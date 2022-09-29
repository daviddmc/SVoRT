import torch
from transform import RigidTransform
import nibabel as nib 
import numpy as np 

def save_volume(fname, data, res):
    data = data[0,0].detach().cpu().numpy().transpose(2, 1, 0)
    w, h, d = data.shape
    affine = np.eye(4)
    affine[:3, :3] *= res
    affine[:3, -1] = - np.array([(w-1)/2., (h-1)/2., (d-1)/2.]) * res
    img = nib.Nifti1Image(data, affine)
    img.header.set_xyzt_units(2)
    img.header.set_qform(affine, code='aligned')
    img.header.set_sform(affine, code='scanner')
    nib.save(img, fname)

def save_stack(slices, transforms, res_s, s_thick, scale, dtype, fname):
    data = (slices * scale).squeeze(1).cpu().numpy().astype(dtype).transpose(2, 1, 0)
    w, h, d = data.shape
    affine = np.eye(4)
    R = transforms[0].matrix()[0, :, :-1].cpu().numpy()
    T = transforms[0].matrix()[0, :, -1].cpu().numpy()
    T[0] -= (w-1)/2 * res_s
    T[1] -= (h-1)/2 * res_s
    T = R @ T.reshape(3,1)
    R = R @ np.diag([res_s, res_s, s_thick])

    affine[:3, :] = np.concatenate((R, T), -1)
    img = nib.Nifti1Image(data, affine)
    img.header.set_xyzt_units(2)
    img.header.set_qform(affine, code='aligned')
    img.header.set_sform(affine, code='scanner')
    nib.save(img, fname)
    return fname    
    
def load_stack(f, device):
    img = nib.load(f)
    slices = img.get_fdata().astype(np.float32).transpose(2, 1, 0)
    d, h, w = slices.shape
    slices = torch.tensor(slices, device=device).unsqueeze(1)
    res = img.header['pixdim'][1:4]
    affine = img.affine
    if np.any(np.isnan(affine)):
        affine = img.get_qform()
    
    R = affine[:3,:3]
    if np.linalg.det(R) < 0:
        print('warning: det(R) < 0')
    T = affine[:3,-1:]
    R = R @ np.linalg.inv(np.diag(res))
    
    T0 = np.array([(w-1)/2 * res[0], (h-1)/2 * res[1], 0])
    T = np.linalg.inv(R) @ T + T0.reshape(3, 1)
    
    tz = torch.arange(0, d, device=device, dtype=torch.float32) * res[2] + T[2].item()
    tx = torch.ones_like(tz) * T[0].item()
    ty = torch.ones_like(tz) * T[1].item()
    t = torch.stack((tx, ty, tz), -1).view(-1, 3, 1)
    R = torch.tensor(R, device=device).unsqueeze(0).expand(d, -1, -1)
    transforms = RigidTransform(torch.cat((R, t), -1).to(torch.float32), trans_first=True)
    res_s = (res[0] + res[1]) / 2
    s_thick = res[2]
    return slices, transforms, res_s, s_thick