import os
import torch
import torch.nn.functional as F
import nibabel as nib 
import numpy as np 
import collections


def bias_field(volume, bias_size, bias_sigma):
    if isinstance(bias_size, collections.abc.Iterable):
        bias_size = np.random.randint(bias_size[0], bias_size[-1]+1) 
    if isinstance(bias_sigma, collections.abc.Iterable):
        bias_sigma = np.random.uniform(bias_sigma[0], bias_sigma[-1])
    bias = torch.randn((1, 1, bias_size, bias_size, bias_size), dtype=volume.dtype, device=volume.device) * bias_sigma
    bias = F.interpolate(bias, size=volume.shape[-3:], mode='trilinear', align_corners=False)
    volume *= torch.exp(bias)
    return volume

def gamma_transform(volume, g, q, threshold):
    if isinstance(g, collections.abc.Iterable):
        g = np.random.uniform(g[0], g[-1])
    if isinstance(q, collections.abc.Iterable):
        q = np.random.uniform(q[0], q[-1])
    vq = torch.quantile(volume[volume > threshold], q)
    volume = (volume / vq)**g
    threshold = (threshold / vq)**g
    return volume, threshold, vq

class RegisteredFetaDataset:
    def __init__(self, is_test, cfg):
        self.threshold = 150
        self.bias_size = cfg['bias_size']
        self.bias_sigma = cfg['bias_sigma']
        self.gamma = cfg['gamma']
        self.quantile = cfg['quantile']
        self.device = torch.device(cfg['device'])
        self.data_dir = cfg['data_dir']
        self.is_test = is_test
        self.test_set = cfg['test_set']
        self.files = []
        for n in range(1, 81):
            if (not self.is_test) and (n in self.test_set):
                continue
            if self.is_test and (n not in self.test_set):
                continue
            self.files.append((os.path.join(self.data_dir, 'reg%d.nii.gz' % n), os.path.join(self.data_dir, 'seg%d.nii.gz' % n)))
        self.idx = 0

    def __len__(self):
        return len(self.files)

    def get_data(self):
        if self.is_test:
            f = self.files[self.idx]
            self.idx += 1
        else:
            f = self.files[np.random.choice(len(self.files))]

        f, f_seg = f
        volume = nib.load(f).get_fdata()
        volume = torch.from_numpy(volume[None, None, 10:-10, 20:-20,15:-15])
        volume = volume.contiguous().to(dtype=torch.float32, device=self.device)
        seg = nib.load(f_seg).get_fdata()
        seg = torch.from_numpy(seg[None, None, 10:-10, 20:-20,15:-15])
        seg = seg.contiguous().to(device=self.device)

        mask = (seg > 0).float()
        if np.random.rand() > 0.7:
            volume *= mask

        res = 0.8
        data =  {'resolution': res, 'mask': mask}
        if self.is_test:
            volume, threshold, vq = gamma_transform(volume, 1, self.quantile, self.threshold)
        else:
            volume = bias_field(volume, self.bias_size, self.bias_sigma)
            volume, threshold, vq = gamma_transform(volume, self.gamma, self.quantile, self.threshold)

        data['threshold'] = threshold
        data['scale'] = vq
        data['volume'] = volume

        return data
