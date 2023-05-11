import scipy.io as sio 
import numpy as np 
from scipy.spatial.transform import Rotation 
from scipy.interpolate import interp1d
import os

if __name__ == '__main__':
    folder = "./keypoint"
    traj_rot = []
    traj_trans = []
    for f in os.listdir(folder):
        if not f.endswith('.mat'): continue
        print(f)
        joint_coord = sio.loadmat(os.path.join(folder, f))['joint_coord'].astype(np.float32)
        joint_coord = joint_coord[np.all(joint_coord > 0, (1, 2))]
    
        eye_l = joint_coord[..., 7]
        eye_r = joint_coord[..., 8]
        neck = (joint_coord[..., 11] + joint_coord[..., 12]) / 2
        
        origin = (eye_l + eye_r + neck) / 3
        
        x_vec = eye_l - eye_r
        x_vec = x_vec / np.linalg.norm(x_vec, ord=2, axis=-1, keepdims=True)
        
        neck_eye_l = neck - eye_l
        y_vec = np.cross(x_vec, neck_eye_l)
        y_vec = y_vec / np.linalg.norm(y_vec, ord=2, axis=-1, keepdims=True)
        
        z_vec = np.cross(x_vec, y_vec)
        z_vec = z_vec / np.linalg.norm(z_vec, ord=2, axis=-1, keepdims=True)
        
        R = np.stack([x_vec, y_vec, z_vec], -1)
        R = R @ R[0].T[None]
        t = (origin - origin[[0]]) * 3 # in mm

        R = Rotation.from_matrix(R).as_euler('xyz') # in rad

        R1 = R[::2]
        interp_func = interp1d(np.arange(R1.shape[0]), R1, kind='linear', axis=0, fill_value="extrapolate", assume_sorted=True)
        traj_rot.append((interp_func, R1.shape[0]-1, 7))
        R2 = R[1::2]
        interp_func = interp1d(np.arange(R2.shape[0]), R2, kind='linear', axis=0, fill_value="extrapolate", assume_sorted=True)
        traj_rot.append((interp_func, R2.shape[0]-1, 7))

        t1 = t[::2]
        interp_func = interp1d(np.arange(t1.shape[0]), t1, kind='linear', axis=0, fill_value="extrapolate", assume_sorted=True)
        traj_trans.append((interp_func, t1.shape[0]-1, 7))
        t2 = t[1::2]
        interp_func = interp1d(np.arange(t2.shape[0]), t2, kind='linear', axis=0, fill_value="extrapolate", assume_sorted=True)
        traj_trans.append((interp_func, t2.shape[0]-1, 7))

    np.save('traj.npy', (traj_rot, traj_trans))