import torch
import numpy as np
from .transform_convert import axisangle2mat, mat2axisangle
from scipy.spatial.transform import Rotation

class RigidTransform(object):
    def __init__(self, data, trans_first=True, device=None):
        self.trans_first = trans_first
        if device is not None:
            data = data.to(device)
        if data.shape[1] == 6: # parameter
            self._axisangle = data
            self._matrix = None
        elif data.shape[1] == 3: # matrix
            self._axisangle = None
            self._matrix = data
        else:
            raise Exception('Unknown format for rigid transform!')

    def matrix(self, trans_first=True):
        if self._matrix is not None:
            mat = self._matrix
        else:
            mat = axisangle2mat(self._axisangle)
        if self.trans_first == True and trans_first == False:
            mat = mat_first2last(mat)
        elif self.trans_first == False and trans_first == True:
            mat = mat_last2first(mat)
        return mat

    def axisangle(self, trans_first=True):
        if self._axisangle is not None:
            ax = self._axisangle
        else:
            ax = mat2axisangle(self._matrix)
        if self.trans_first == True and trans_first == False:
            ax = ax_first2last(ax)
        elif self.trans_first == False and trans_first == True:
            ax = ax_last2first(ax)
        return ax

    def inv(self):
        mat = self.matrix(trans_first=True)
        R = mat[:, :, :3]
        t = mat[:, :, 3:]
        mat = torch.cat((R.transpose(-2, -1), -torch.matmul(R, t)), -1)
        return RigidTransform(mat, trans_first=True)

    def compose(self, other):
        mat1 = self.matrix(trans_first=True)
        mat2 = other.matrix(trans_first=True)
        R1 = mat1[:, :, :3]
        t1 = mat1[:, :, 3:]
        R2 = mat2[:, :, :3]
        t2 = mat2[:, :, 3:]
        R = torch.matmul(R1, R2)
        t = t2 + torch.matmul(R2.transpose(-2, -1), t1)
        mat = torch.cat((R, t), -1)
        return RigidTransform(mat, trans_first=True)

    def __getitem__(self, idx):
        if self._axisangle is not None:
            data = self._axisangle[idx]
            if len(data.shape) < 2:
                data = data.unsqueeze(0)
        else:
            data = self._matrix[idx]
            if len(data.shape) < 3:
                data = data.unsqueeze(0)
        return RigidTransform(data, self.trans_first)

    def detach(self):
        if self._axisangle is not None:
            data = self._axisangle.detach()
        else:
            data = self._matrix.detach()
        return RigidTransform(data, self.trans_first)

    @staticmethod
    def cat(transforms):
        matrixs = [t.matrix(trans_first=True) for t in transforms]
        return RigidTransform(torch.cat(matrixs, 0), trans_first=True)

'''helper for RigidTransform'''

def mat_first2last(mat):
    R = mat[:, :, :3]
    t = mat[:, :, 3:]
    t = torch.matmul(R, t)
    mat = torch.cat([R, t], -1)
    return mat
    
def mat_last2first(mat):
    R = mat[:, :, :3]
    t = mat[:, :, 3:]
    t = torch.matmul(R.transpose(-2, -1), t)
    mat = torch.cat([R, t], -1)
    return mat

def ax_first2last(axisangle):
    mat = axisangle2mat(axisangle)
    mat = mat_first2last(mat)
    return mat2axisangle(mat)

def ax_last2first(axisangle):
    mat = axisangle2mat(axisangle)
    mat = mat_last2first(mat)
    return mat2axisangle(mat)

def mat_update_resolution(mat, res_from, res_to):
    assert mat.dim() == 3
    fac = torch.ones_like(mat[:1, :1])
    fac[..., 3] = res_from/res_to
    return mat * fac

def ax_update_resolution(ax, res_from, res_to):
    assert ax.dim() == 2
    fac = torch.ones_like(ax[:1])
    fac[:, 3:] = res_from/res_to
    return ax * fac

# random angle
def random_angle(n, restricted, device):
    a = 2*np.pi*np.random.rand(n)
    b = np.arccos(2 *np.random.rand(n) - 1)
    if restricted:
        c = np.pi * np.random.rand(n)
    else:
        c = np.pi * (2 *np.random.rand(n) - 1)

    R = Rotation.from_euler("ZXZ", np.stack([a, b, c], -1))   
    R = R.as_rotvec()
    return torch.from_numpy(R).to(dtype=torch.float32, device=device)

def random_trans(n, T_range, device):
    if not isinstance(T_range, (list, tuple)):
        T_range = [T_range, T_range, T_range]
    else:
        assert(len(T_range) == 3)
    tx = (torch.rand(n, device=device)-0.5) * T_range[0]
    ty = (torch.rand(n, device=device)-0.5) * T_range[1]
    tz = (torch.rand(n, device=device)-0.5) * T_range[2]
    return torch.stack([tx,ty,tz], -1)

'''misc'''

def mat2euler(mat):
    TOL = 0.000001
    TX = mat[:, 0, 3]
    TY = mat[:, 1, 3]
    TZ = mat[:, 2, 3]

    tmp = torch.asin(-mat[:, 0, 2])
    mask = torch.cos(tmp).abs() <= TOL
    RX = torch.atan2(mat[:, 1, 2], mat[:, 2, 2])
    RY = tmp
    RZ = torch.atan2(mat[:, 0, 1], mat[:, 0, 0])
    RX[mask] = torch.atan2(-mat[:,0,2]*mat[:,1,0], -mat[:, 0,2]*mat[:, 2,0])[mask]
    RZ[mask] = 0

    RX *= 180/np.pi
    RY *= 180/np.pi
    RZ *= 180/np.pi

    return torch.stack((TX, TY, TZ, RX, RY, RZ), -1)

def euler2mat(p):
    tx = p[:, 0]
    ty = p[:, 1]
    tz = p[:, 2]

    rx = p[:, 3]
    ry = p[:, 4]
    rz = p[:, 5]

    M_PI = np.pi
    cosrx = torch.cos(rx*(M_PI/180.0))
    cosry = torch.cos(ry*(M_PI/180.0))
    cosrz = torch.cos(rz*(M_PI/180.0))
    sinrx = torch.sin(rx*(M_PI/180.0))
    sinry = torch.sin(ry*(M_PI/180.0))
    sinrz = torch.sin(rz*(M_PI/180.0))

    mat = torch.eye(4, device=p.device)
    mat = mat.reshape((1, 4, 4)).repeat(p.shape[0], 1, 1)
    
    mat[:, 0,0] = cosry*cosrz
    mat[:, 0,1] = cosry*sinrz
    mat[:, 0,2] = -sinry
    mat[:, 0,3] = tx

    mat[:,1,0] = (sinrx*sinry*cosrz-cosrx*sinrz)
    mat[:,1,1] = (sinrx*sinry*sinrz+cosrx*cosrz)
    mat[:, 1,2] = sinrx*cosry
    mat[:, 1,3] = ty

    mat[:, 2,0] = (cosrx*sinry*cosrz+sinrx*sinrz)
    mat[:, 2,1] = (cosrx*sinry*sinrz-sinrx*cosrz)
    mat[:, 2,2] = cosrx*cosry
    mat[:, 2,3] = tz
    mat[:, 3,3] = 1.0

    return mat


def point2mat(p):
    p = p.view(-1, 3, 3)
    p1 = p[:, 0]
    p2 = p[:, 1]
    p3 = p[:, 2]
    v1 = p3 - p1
    v2 = p2 - p1

    nz = torch.cross(v1, v2, -1)
    ny = torch.cross(nz, v1, -1)
    nx = v1

    R = torch.stack((nx, ny, nz), -1)
    R = R / torch.linalg.norm(R, ord=2, dim=-2, keepdim=True)

    T = torch.matmul(R.transpose(-2, -1), p2.unsqueeze(-1))

    return torch.cat((R, T), -1)


def mat2point(mat, sx, sy, rs):
    p1 = torch.tensor([-(sx- 1) / 2 * rs, -(sy - 1) / 2 * rs, 0]).to(dtype=mat.dtype, device=mat.device)
    p2 = torch.tensor([0, 0, 0]).to(dtype=mat.dtype, device=mat.device)
    p3 = torch.tensor([(sx - 1) / 2 * rs, -(sy - 1) / 2 * rs, 0]).to(dtype=mat.dtype, device=mat.device)
    p = torch.stack((p1, p2, p3), 0)
    p = p.unsqueeze(0).unsqueeze(-1) # 1x3x3x1
    R = mat[:, :, :-1].unsqueeze(1) # nx1x3x3
    T = mat[:, :, -1:].unsqueeze(1) # nx1x3x1
    p = torch.matmul(R, p + T)
    return p.view(-1, 9)