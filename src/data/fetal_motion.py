import scipy.io as sio 
import numpy as np 
from scipy.spatial.transform import Rotation 
import os
from transform import RigidTransform
import torch

_traj_rot = None
_traj_trans = None

def get_trajectory():
    global _traj_rot, _traj_trans
    if _traj_rot is not None:
        return _traj_rot, _traj_trans
    _traj_rot, _traj_trans = np.load('../dataset/traj.npy', allow_pickle=True)
    return _traj_rot, _traj_trans

def sample_motion(ts, device):
    trajs_rot, trajs_trans = get_trajectory()
    # rotation
    traj, T, dT = trajs_rot[np.random.choice(len(trajs_rot))]
    t0 = np.random.uniform(0, T - ts[-1] / dT)
    R = traj(t0 + ts / dT)
    R = R[:, np.random.permutation(3)] # random permute
    R = R * (2 * (np.random.rand(1, 3) < 0.5) - 1) # random flip
    R = Rotation.from_euler('xyz', R).as_matrix()
    # translation
    traj, T, dT = trajs_trans[np.random.choice(len(trajs_trans))]
    t0 = np.random.uniform(0, T - ts[-1] / dT)
    trans = traj(t0 + ts / dT)
    trans = trans[:, np.random.permutation(3)]
    trans = trans * (2 * (np.random.rand(1, 3) < 0.5) - 1)

    R = torch.tensor(R, dtype=torch.float32, device=device)
    trans = torch.tensor(trans, dtype=torch.float32, device=device)

    R = torch.matmul(R, R[0].transpose(-2, -1))
    trans = trans - trans[0]
    
    transforms_motion = RigidTransform(torch.cat((R, trans.unsqueeze(-1)), -1), trans_first=False)
    
    return transforms_motion