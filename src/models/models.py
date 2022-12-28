import random
import torch
import torch.nn as nn
from .transformer import SVRtransformer, SRRtransformer, SVRtransformerV2
from .reconstruction import PSFreconstruction, SRR
from transform import (
    RigidTransform,
    mat_update_resolution,
    ax_update_resolution,
    mat2axisangle,
    point2mat,
    mat2point,
)


class SVoRT(nn.Module):
    def __init__(self, n_iter=3, iqa=True, vol=True, pe=True):
        super().__init__()
        self.n_iter = n_iter
        self.vol = vol
        self.pe = pe
        self.iqa = iqa and vol
        self.attn = None
        self.iqa_score = None

        svrnet_list = []
        for i in range(self.n_iter):
            svrnet_list.append(
                SVRtransformer(
                    n_res=50,
                    n_layers=4,
                    n_head=4 * 2,
                    d_in=9 + 2,
                    d_out=9,
                    d_model=256 * 2,
                    d_inner=512 * 2,
                    dropout=0.0,
                    res_d_in=4 if (i > 0 and self.vol) else 3,
                )
            )
        self.svrnet = nn.ModuleList(svrnet_list)
        if iqa:
            self.srrnet = SRRtransformer(
                n_res=34,
                n_layers=4,
                n_head=4,
                d_in=8,
                d_out=1,
                d_model=256,
                d_inner=512,
                dropout=0.0,
            )

    def forward(self, data):

        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "interp_psf": False,
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "s_thick": data["slice_thickness"],
            "volume_shape": data["volume_shape"],
        }

        transforms = RigidTransform(data["transforms"])
        stacks = data["stacks"]
        positions = data["positions"]

        thetas = []
        volumes = []
        trans = []

        if not self.pe:
            transforms = RigidTransform(transforms.axisangle() * 0)
            positions = positions * 0 + data["slice_thickness"]

        theta = mat2point(
            transforms.matrix(), stacks.shape[-1], stacks.shape[-2], params["res_s"]
        )
        volume = None

        mask_stacks = None

        for i in range(self.n_iter):
            theta, attn = self.svrnet[i](
                theta,
                stacks,
                positions,
                None if ((volume is None) or (not self.vol)) else volume.detach(),
                params,
            )

            thetas.append(theta)

            _trans = RigidTransform(point2mat(theta))
            trans.append(_trans)

            with torch.no_grad():
                mat = mat_update_resolution(
                    _trans.matrix().detach(), 1, params["res_r"]
                )
                volume = PSFreconstruction(mat, stacks, mask_stacks, None, params)
                ax = mat2axisangle(_trans.matrix())
                ax = ax_update_resolution(ax, 1, params["res_s"])
            if self.iqa:
                volume, iqa_score = self.srrnet(
                    ax, mat, stacks, volume, params, positions
                )
                self.iqa_score = iqa_score.detach()
            volumes.append(volume)

        self.attn = attn.detach()

        return trans, volumes, thetas


def build_attn_mask(n_slices, fill_value, dtype, device):
    n_stack = len(n_slices)
    attn_mask = torch.zeros(
        (sum(n_slices) + n_stack, sum(n_slices) + n_stack), dtype=dtype, device=device
    )
    attn_mask[:, :n_stack] = fill_value
    i_slice = 0
    for i_stack, n_slice in enumerate(n_slices):
        attn_mask[i_stack, n_stack + i_slice : n_stack + i_slice + n_slice] = fill_value
        attn_mask[
            n_stack + i_slice : n_stack + i_slice + n_slice,
            n_stack + i_slice : n_stack + i_slice + n_slice,
        ] = fill_value
        i_slice += n_slice
    return attn_mask


class SVoRTv2(nn.Module):
    def __init__(self, n_iter=4, iqa=True, vol=True, pe=True):
        super().__init__()
        self.vol = vol
        self.pe = pe
        self.iqa = iqa and vol
        self.attn = None
        self.iqa_score = None
        self.n_iter = n_iter

        self.svrnet1 = SVRtransformerV2(
            n_layers=4,
            n_head=4 * 2,
            d_in=9 + 2,
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=1,
        )

        self.svrnet2 = SVRtransformerV2(
            n_layers=4 * 2,
            n_head=4 * 2,
            d_in=9 + 2,
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=2,
        )

        if iqa:
            self.srr = SRR(n_iter=2, use_CG=True)

    def forward(self, data, n_iter=None):

        if n_iter is None:
            n_iter = self.n_iter

        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "interp_psf": False,
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "s_thick": data["slice_thickness"],
            "volume_shape": data["volume_shape"],
        }

        transforms = RigidTransform(data["transforms"])
        stacks = data["stacks"]
        positions = data["positions"]
        attn_mask = None

        thetas = []
        volumes = []
        trans = []

        if not self.pe:
            transforms = RigidTransform(transforms.axisangle() * 0)
            positions = positions * 0 + data["slice_thickness"]

        theta = mat2point(
            transforms.matrix(), stacks.shape[-1], stacks.shape[-2], params["res_s"]
        )
        volume = None

        mask_stacks = None

        default_grad = torch.is_grad_enabled()
        if n_iter > 1:
            iter_grad = random.randint(1, n_iter - 1)
        else:
            iter_grad = 0

        for i in range(n_iter):
            if i != 0 and i != iter_grad:
                torch.set_grad_enabled(False)
            svrnet = self.svrnet2 if i else self.svrnet1
            theta, iqa_score, attn = svrnet(
                theta,
                stacks,
                positions,
                None if ((volume is None) or (not self.vol)) else volume.detach(),
                params,
                attn_mask,
            )

            thetas.append(theta)

            _trans = RigidTransform(point2mat(theta))

            with torch.no_grad():
                if (i == 0 or i == iter_grad) and "transforms_gt" in data:
                    # adjust for GT
                    diff = (
                        RigidTransform(data["transforms_gt"])
                        .compose(_trans.inv())
                        .axisangle()
                        .detach()
                    )
                    err_R = torch.sum(diff[:, :3] ** 2, -1)
                    err_T = torch.sum(diff[:, 3:] ** 2, -1)
                    err = 1000.0 * err_R + err_T
                    _, min_idx = torch.topk(err, 5, largest=False, sorted=False)
                    diff = diff[min_idx, :].mean(0, keepdim=True)
                else:
                    diff = None

            if diff is not None:
                _trans = RigidTransform(diff).compose(_trans)

            trans.append(_trans)

            with torch.no_grad():
                mat = mat_update_resolution(
                    _trans.matrix().detach(), 1, params["res_r"]
                )
                volume = PSFreconstruction(mat, stacks, mask_stacks, None, params)
            if self.iqa:
                volume = self.srr(
                    mat, stacks, volume, params, iqa_score.view(-1, 1, 1, 1)
                )
                self.iqa_score = iqa_score.detach()
            volumes.append(volume)
            torch.set_grad_enabled(default_grad)

        self.attn = attn.detach()

        return trans, volumes, thetas
