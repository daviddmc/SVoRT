import torch
import torch.nn as nn
from .transformer import SVRtransformer, SRRtransformer
from .reconstruction import PSFreconstruction
from transform import RigidTransform, mat_update_resolution, ax_update_resolution, mat2axisangle, point2mat, mat2point


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
                  n_layers=4, n_head=4*2, d_in=9+2, d_out=9, d_model=256*2, d_inner=512*2, dropout=0.1, 
                  res_d_in=4 if (i > 0 and self.vol) else 3
                  )
            )
        self.svrnet = nn.ModuleList(svrnet_list)
        if iqa:
            self.srrnet = SRRtransformer(n_res=34, n_layers=4, n_head=4, d_in=8, d_out=1, d_model=256, d_inner=512, dropout=0.1)

    def forward(self, data):

        params = {
            'psf': data['psf_rec'], 
            'slice_shape':data['slice_shape'], 
            'interp_psf':False, 
            'res_s':data['resolution_slice'], 
            'res_r':data['resolution_recon'], 
            's_thick': data['slice_thickness'],
            'volume_shape':data['volume_shape'],
            }

        transforms = RigidTransform(data['transforms'])
        stacks = data['stacks']
        positions = data['positions']

        thetas = []
        volumes = []
        trans = []
        
        if not self.pe:
            transforms = RigidTransform(transforms.axisangle() * 0)
            positions = positions*0 + data['slice_thickness']
        
        theta = mat2point(transforms.matrix(), stacks.shape[-1], stacks.shape[-2], params['res_s'])
        volume = None

        mask_stacks = None 

        for i in range(self.n_iter):
            theta, attn = self.svrnet[i](theta, stacks, positions, None if ((volume is None) or (not self.vol)) else volume.detach(), params)
            
            thetas.append(theta)

            _trans = RigidTransform(point2mat(theta))
            trans.append(_trans)

            with torch.no_grad():
                mat = mat_update_resolution(_trans.matrix().detach(), 1, params['res_r'])
                volume = PSFreconstruction(mat, stacks, mask_stacks, None, params)
                ax = mat2axisangle(_trans.matrix())
                ax = ax_update_resolution(ax, 1, params['res_s'])
            if self.iqa:
                volume, iqa_score = self.srrnet(ax, mat, stacks, volume, params, positions)
                self.iqa_score = iqa_score.detach()
            volumes.append(volume)
            
        self.attn = attn.detach()

        return trans, volumes, thetas


