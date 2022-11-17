import torch
import torch.nn.functional as F
from transform import random_angle, RigidTransform, mat_update_resolution
from slice_acquisition import slice_acquisition
from .fetal_motion import sample_motion
import numpy as np


def interleave_index(N, n_i):
    idx = [None] * N
    t = 0
    for i in range(n_i):
        j = i
        while j < N:
            idx[j] = t
            t += 1
            j += n_i
    return idx


def get_PSF(
    r_max=None, res_ratio=(1, 1, 3), threshold=1e-3, device=torch.device("cpu")
):
    sigma_x = 1.2 * res_ratio[0] / 2.3548
    sigma_y = 1.2 * res_ratio[1] / 2.3548
    sigma_z = res_ratio[2] / 2.3548
    if r_max is None:
        r_max = max(int(2 * r + 1) for r in (sigma_x, sigma_y, sigma_z))
        r_max = max(r_max, 4)
    x = torch.linspace(-r_max, r_max, 2 * r_max + 1, dtype=torch.float32, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing="ij")
    psf = torch.exp(
        -0.5
        * (
            grid_x**2 / sigma_x**2
            + grid_y**2 / sigma_y**2
            + grid_z**2 / sigma_z**2
        )
    )
    psf[psf.abs() < threshold] = 0
    rx = torch.nonzero(psf.sum((0, 1)) > 0)[0, 0].item()
    ry = torch.nonzero(psf.sum((0, 2)) > 0)[0, 0].item()
    rz = torch.nonzero(psf.sum((1, 2)) > 0)[0, 0].item()
    psf = psf[
        rz : 2 * r_max + 1 - rz, ry : 2 * r_max + 1 - ry, rx : 2 * r_max + 1 - rx
    ].contiguous()
    psf = psf / psf.sum()
    return psf


def init_stack_transforms(n_slice, gap, restricted, txy, device):
    angle = random_angle(1, restricted, device).expand(n_slice, -1)
    tz = (
        torch.arange(0, n_slice, device=device, dtype=torch.float32)
        - (n_slice - 1) / 2.0
    ) * gap
    if txy:
        tx = torch.ones_like(tz) * np.random.uniform(-txy, txy)
        ty = torch.ones_like(tz) * np.random.uniform(-txy, txy)
    else:
        tx = ty = torch.zeros_like(tz)
    t = torch.stack((tx, ty, tz), -1)
    return RigidTransform(torch.cat((angle, t), -1), trans_first=True)


def reset_transform(transform):
    transform = transform.axisangle()
    transform[:, :-1] = 0
    transform[:, -1] -= transform[:, -1].mean()
    return RigidTransform(transform)


class Scanner:
    def __init__(self, kwargs):
        self.resolution_slice_fac = kwargs["resolution_slice_fac"]
        self.resolution_slice_max = kwargs["resolution_slice_max"]
        self.slice_thickness = kwargs["slice_thickness"]
        self.gap = kwargs["gap"]
        self.min_num_stack = kwargs["min_num_stack"]
        self.max_num_stack = kwargs["max_num_stack"]
        self.max_num_slices = kwargs.get("max_num_slices", None)
        self.noise_sigma = kwargs["noise_sigma"]
        self.TR = kwargs["TR"]
        self.prob_void = kwargs["prob_void"]
        self.slice_size = kwargs.get("slice_size", None)
        self.resolution_recon = kwargs.get("resolution_recon", None)
        self.restrict_transform = kwargs.get("restrict_transform", False)
        self.txy = kwargs.get("txy", 0)

    def get_resolution(self, data):
        resolution = data["resolution"]
        if hasattr(self.resolution_slice_fac, "__len__"):
            resolution_slice = np.random.uniform(
                self.resolution_slice_fac[0] * resolution,
                min(
                    self.resolution_slice_fac[-1] * resolution,
                    self.resolution_slice_max,
                ),
            )
        else:
            resolution_slice = self.resolution_slice_fac * resolution
        if self.resolution_recon is not None:
            data["resolution_recon"] = self.resolution_recon
        else:
            data["resolution_recon"] = np.random.uniform(resolution, resolution_slice)
        data["resolution_slice"] = resolution_slice
        if hasattr(self.slice_thickness, "__len__"):
            data["slice_thickness"] = np.random.uniform(
                self.slice_thickness[0], self.slice_thickness[-1]
            )
        else:
            data["slice_thickness"] = self.slice_thickness
        if self.gap is None:
            data["gap"] = data["slice_thickness"]
        elif hasattr(self.gap, "__len__"):
            data["gap"] = np.random.uniform(
                self.gap[0], min(self.gap[-1], data["slice_thickness"])
            )
        else:
            data["gap"] = self.gap
        return data

    def add_noise(self, slices, threshold):
        if (not hasattr(self.noise_sigma, "__len__")) and self.noise_sigma == 0:
            return slices
        mask = slices > threshold
        masked = slices[mask]
        sigma = np.random.uniform(self.noise_sigma[0], self.noise_sigma[-1])
        noise1 = torch.randn_like(masked) * sigma
        noise2 = torch.randn_like(masked) * sigma
        slices[mask] = torch.sqrt((masked + noise1) ** 2 + noise2**2)
        return slices

    def signal_void(self, slices):

        idx = torch.rand(slices.shape[0], device=slices.device) < self.prob_void
        n = idx.sum()
        if n > 0:
            h, w = slices.shape[-2:]
            y = torch.linspace(-(h - 1) / 2, (h - 1) / 2, h, device=slices.device)
            x = torch.linspace(-(w - 1) / 2, (w - 1) / 2, w, device=slices.device)
            yc = (torch.rand(n, device=slices.device) - 0.5) * (h - 1)
            xc = (torch.rand(n, device=slices.device) - 0.5) * (w - 1)

            y = y.view(1, -1, 1) - yc.view(-1, 1, 1)
            x = x.view(1, 1, -1) - xc.view(-1, 1, 1)

            theta = 2 * np.pi * torch.rand((n, 1, 1), device=slices.device)
            c = torch.cos(theta)
            s = torch.sin(theta)
            x, y = c * x - s * y, s * x + c * y

            a = 5 + torch.rand_like(theta) * 10
            A = torch.rand_like(theta) * 0.5 + 0.5
            sx = torch.rand_like(theta) * 15 + 1
            sy = a**2 / sx

            sx = -0.5 / sx**2
            sy = -0.5 / sy**2

            mask = 1 - A * torch.exp(sx * x**2 + sy * y**2)
            slices[idx, 0] *= mask
        return slices

    def sample_time(self, n_slice):
        TR = np.random.uniform(self.TR[0], self.TR[1])
        return np.arange(n_slice) * TR

    def scan(self, data):
        data = self.get_resolution(data)
        res = data["resolution"]
        res_r = data["resolution_recon"]
        res_s = data["resolution_slice"]
        s_thick = data["slice_thickness"]
        gap = data["gap"]
        device = data["volume"].device
        if res_r != res:
            grids = []
            for i in range(3):
                size_new = int(data["volume"].shape[i + 2] * res / res_r)
                grid_max = (
                    (size_new - 1) * res_r / (data["volume"].shape[i + 2] - 1) / res
                )
                grids.append(
                    torch.linspace(-grid_max, grid_max, size_new, device=device)
                )
            grid = torch.stack(
                torch.meshgrid(*grids, indexing="ij")[::-1], -1
            ).unsqueeze_(0)
            volume_gt = F.grid_sample(data["volume"], grid, align_corners=True)
        else:
            volume_gt = data["volume"].clone()
        data["volume_gt"] = volume_gt

        psf_acq = get_PSF(
            res_ratio=(res_s / res, res_s / res, s_thick / res), device=device
        )
        psf_rec = get_PSF(
            res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r), device=device
        )
        data["psf_rec"] = psf_rec
        data["psf_acq"] = psf_acq

        vs = data["volume"].shape
        if self.slice_size is None:
            ss = int(
                np.sqrt((vs[-1] ** 2 + vs[-2] ** 2 + vs[-3] ** 2) / 2.0) * res / res_s
            )
            ss = int(np.ceil(ss / 32.0) * 32)
        else:
            ss = self.slice_size
        ns = int(max(vs) * res / gap) + 2

        stacks = []
        transforms = []
        transforms_gt = []
        positions = []

        max_num_stack = np.random.randint(self.min_num_stack, self.max_num_stack + 1)
        while True:
            # stack transformation
            transform_init = init_stack_transforms(
                ns, gap, self.restrict_transform, self.txy, device
            )
            ts = self.sample_time(ns)
            transform_motion = sample_motion(ts, device)
            # interleaved acquisition
            interleave_idx = interleave_index(
                ns, np.random.randint(2, int(np.sqrt(ns)) + 1)
            )
            transform_motion = transform_motion[interleave_idx]
            # apply motion
            transform_target = transform_motion.compose(transform_init)
            # sample slices
            mat = mat_update_resolution(transform_target.matrix(), 1, res)
            slices = slice_acquisition(
                mat,
                data["volume"],
                None,
                None,
                psf_acq,
                (ss, ss),
                res_s / res,
                False,
                False,
            )
            slices_no_psf = slice_acquisition(
                mat,
                data["mask"],
                None,
                None,
                get_PSF(0, device=device),
                (ss, ss),
                res_s / res,
                False,
                False,
            )
            # remove zeros
            nnz = slices_no_psf.sum((1, 2, 3))
            idx = nnz > (nnz.max() * np.random.uniform(0.1, 0.3))
            if idx.sum() == 0:
                print("empty")
                continue
            else:
                nz = torch.nonzero(idx)
                idx[nz[0, 0] : nz[-1, 0]] = True
            slices = slices[idx]
            transform_init = transform_init[idx]
            transform_init = reset_transform(transform_init)
            transform_target = transform_target[idx]
            # artifacts
            slices = self.add_noise(slices, data["threshold"])
            slices = self.signal_void(slices)
            # append stack
            if (
                self.max_num_slices is not None
                and sum(st.shape[0] for st in stacks) + slices.shape[0]
                >= self.max_num_slices
            ):
                break
            stacks.append(slices)
            transforms.append(transform_init)
            transforms_gt.append(transform_target)
            positions.append(
                torch.arange(slices.shape[0], dtype=slices.dtype, device=device)
                - slices.shape[0] // 2
            )
            if len(stacks) >= max_num_stack:
                break
        # add stack index
        stacks_ids = np.random.choice(20, len(stacks), replace=False)
        positions = torch.cat(
            [
                torch.stack((positions[i], torch.full_like(positions[i], s_i)), -1)
                for i, s_i in enumerate(stacks_ids)
            ],
            0,
        )
        stacks = torch.cat(stacks, 0)
        transforms = RigidTransform.cat(transforms)
        transforms_gt = RigidTransform.cat(transforms_gt)

        data["slice_shape"] = (ss, ss)
        data["volume_shape"] = volume_gt.shape[-3:]
        data["stacks"] = stacks
        data["positions"] = positions
        data["transforms"] = transforms.matrix()
        data["transforms_gt"] = transforms_gt.matrix()
        data.pop("volume")

        return data
