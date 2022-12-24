import os
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import collections


def bias_field(volume, bias_size, bias_sigma):
    if isinstance(bias_size, collections.abc.Iterable):
        bias_size = np.random.randint(bias_size[0], bias_size[-1] + 1)
    if isinstance(bias_sigma, collections.abc.Iterable):
        bias_sigma = np.random.uniform(bias_sigma[0], bias_sigma[-1])
    bias = (
        torch.randn(
            (1, 1, bias_size, bias_size, bias_size),
            dtype=volume.dtype,
            device=volume.device,
        )
        * bias_sigma
    )
    bias = F.interpolate(
        bias, size=volume.shape[-3:], mode="trilinear", align_corners=False
    )
    volume *= torch.exp(bias)
    return volume


def gamma_transform(volume, g, q, threshold):
    if isinstance(g, collections.abc.Iterable):
        g = np.random.uniform(g[0], g[-1])
    if isinstance(q, collections.abc.Iterable):
        q = np.random.uniform(q[0], q[-1])
    vq = torch.quantile(volume[volume > threshold], q)
    volume = (volume / vq) ** g
    threshold = (threshold / vq) ** g
    return volume, threshold, vq


class CombinedDataset:
    def __init__(self, is_test, cfg):
        self.is_test = is_test
        self.datasets = []
        for key in cfg:
            if key in globals():
                self.datasets.append(globals()[key](is_test, cfg))
        self.idx = 0

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def get_data(self):
        if self.is_test:
            L = 0
            for dataset in self.datasets:
                L += len(dataset)
                if self.idx < L:
                    data = dataset.get_data()
            self.idx += 1
        else:
            p = np.array([len(dataset) for dataset in self.datasets])
            p = p / p.sum()
            data = self.datasets[np.random.choice(len(self.datasets), p=p)].get_data()
        return data


class AugmentationDataset:
    def __init__(self, is_test, cfg):
        self.threshold = cfg["threshold"]
        self.bias_size = cfg["bias_size"]
        self.bias_sigma = cfg["bias_sigma"]
        self.gamma = cfg["gamma"]
        self.device = torch.device(cfg["device"])
        self.data_dir = cfg["data_dir"]
        self.is_test = is_test
        self.test_set = cfg["test_set"]
        self.files = self.get_file()
        self.masks = self.get_mask()
        self.idx = 0

    def get_file(self):
        raise NotImplementedError("")

    def get_mask(self):
        raise NotImplementedError("")

    def __len__(self):
        return len(self.files)

    def get_data(self):
        if self.is_test:
            f = self.files[self.idx]
            m = self.masks[self.idx]
            self.idx += 1
        else:
            idx = np.random.choice(len(self.files))
            f = self.files[idx]
            m = self.masks[idx]

        volume = nib.load(f).get_fdata()
        volume = torch.from_numpy(volume[None, None, 10:-10, 20:-20, 15:-15])
        volume = torch.flip(volume.permute(0, 1, 4, 3, 2), (-1, -2))
        volume = volume.contiguous().to(dtype=torch.float32, device=self.device)
        if m is not None:
            seg = nib.load(m).get_fdata()
            seg = torch.from_numpy(seg[None, None, 10:-10, 20:-20, 15:-15])
            seg = torch.flip(seg.permute(0, 1, 4, 3, 2), (-1, -2))
            seg = seg.contiguous().to(device=self.device)
            mask = (seg > 0).float()
            if np.random.rand() > 0.7:
                volume *= mask
        else:
            mask = (volume > 0).float()

        res = 0.8
        data = {"resolution": res, "mask": mask}
        if self.is_test:
            volume, threshold, vq = gamma_transform(volume, 1, 0.99, self.threshold)
        else:
            volume = bias_field(volume, self.bias_size, self.bias_sigma)
            volume, threshold, vq = gamma_transform(
                volume, self.gamma, 0.99, self.threshold
            )

        data["threshold"] = threshold
        data["scale"] = vq
        data["volume"] = volume

        return data


class RegisteredFetaDataset(AugmentationDataset):
    def __init__(self, is_test, cfg):
        cfg = cfg.copy()
        if self.__class__.__name__ in cfg:
            cfg.update(cfg[self.__class__.__name__])
        cfg["threshold"] = 150
        super().__init__(is_test, cfg)

    def get_file(self):
        files = []
        for n in range(1, 81):
            if (not self.is_test) and (n in self.test_set):
                continue
            if self.is_test and (n not in self.test_set):
                continue
            files.append(os.path.join(self.data_dir, "reg%d.nii.gz" % n))
        return files

    def get_mask(self):
        masks = []
        for n in range(1, 81):
            if (not self.is_test) and (n in self.test_set):
                continue
            if self.is_test and (n not in self.test_set):
                continue
            masks.append(os.path.join(self.data_dir, "seg%d.nii.gz" % n))
        return masks


class AtlasDataset(AugmentationDataset):
    def __init__(self, is_test, cfg):
        cfg = cfg.copy()
        if self.__class__.__name__ in cfg:
            cfg.update(cfg[self.__class__.__name__])
        cfg["threshold"] = 1
        super().__init__(is_test, cfg)

    def get_file(self):
        files = []
        ga_list = range(21, 38 + 1)
        for ga in ga_list:
            f_atlas = ("STA%d" % ga) if ga < 36 else ("STA%dexp" % ga)
            f_atlas = os.path.join(self.data_dir, f_atlas + ".nii.gz")
            if (not self.is_test) and (ga in self.test_set):
                continue
            if self.is_test and (ga not in self.test_set):
                continue
            files.append(f_atlas)
        return files

    def get_mask(self):
        return [None] * len(self.files)


class RegisteredDataset(AugmentationDataset):
    def __init__(self, is_test, cfg):
        cfg = cfg.copy()
        if self.__class__.__name__ in cfg:
            cfg.update(cfg[self.__class__.__name__])
        cfg["threshold"] = 10
        super().__init__(is_test, cfg)

    def get_file(self):
        def in_test(f, test_set):
            for key in test_set:
                if key in f:
                    return True
            return False

        files = []
        for f in sorted(os.listdir(self.data_dir)):
            if not (f.endswith(".nii") or f.endswith(".nii.gz")):
                continue
            if (not self.is_test) and (in_test(f, self.test_set)):
                continue
            if self.is_test and (not in_test(f, self.test_set)):
                continue
            files.append(os.path.join(self.data_dir, f))
        return files

    def get_mask(self):
        return [None] * len(self.files)
