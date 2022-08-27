import torch
import torch.nn as nn
import torchvision.models as tvm
from .reconstruction import PSFreconstruction
from transform import RigidTransform, mat_update_resolution, point2mat


class Baseline(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

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

        stacks = data['stacks']
        theta = self.model(stacks)

        with torch.no_grad():
            trans = point2mat(theta)
            mat = mat_update_resolution(trans, 1, params['res_r'])
            volume = PSFreconstruction(mat, stacks, None, None, params)

        return [RigidTransform(trans)], [volume], [theta] 
    

class SVRnet(Baseline):

    def __init__(self):
        model = tvm.vgg16(num_classes=9)
        model.features[0] = nn.Conv2d(1, model.features[0].out_channels, 
            kernel_size=model.features[0].kernel_size, padding=model.features[0].padding)
        super().__init__(model)


# copied from https://github.com/pakheiyeung/PlaneInVol/
class PlaneInVol(Baseline):
    def __init__(self):
        model = Proposed_vgg(make_layers_instance_norm(), num_classes=9, fc_size = 512)
        model.apply(weight_init)
        super().__init__(model)

FILTER_SIZE = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
                
def make_layers_instance_norm(norm=True):
    layers = []
    in_channels = 1
    for v in FILTER_SIZE:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if norm:
                layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class Proposed_vgg(nn.Module):
    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device=''):
        super(Proposed_vgg, self).__init__()

        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))

        self.feature_fc = nn.Sequential(
            nn.Linear(FILTER_SIZE[-2] * 5 * 5, fc_size),
            nn.ReLU(True),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(True),
        )

        self.alpha = nn.Sequential(
                nn.Linear(fc_size, fc_size//2),
                nn.ReLU(True),
                nn.Linear(fc_size//2, fc_size//2),
                nn.Sigmoid()
                )
        
        self.beta = nn.Sequential(
                nn.Linear(fc_size, fc_size//2),
                nn.ReLU(True),
                nn.Linear(fc_size//2, fc_size//2),
                nn.Sigmoid()
                )
        
        self.prediction = nn.Sequential(
            nn.Conv2d(fc_size*2, fc_size, kernel_size=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(fc_size, fc_size, kernel_size=1, padding=0),
            nn.ReLU(True),
        )

        self.pt = nn.Linear(fc_size, num_classes)
        
    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        B,C,H,W = x.size()
        
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        alpha = self.alpha(x)
        beta = self.beta(x)
        attention = torch.matmul(alpha, beta.permute(1,0))
        attention_repeat = torch.unsqueeze(attention,-1).repeat(1, 1, self.fc_size)
        attention_sum = attention.sum(1, keepdim=True).repeat(1, self.fc_size)

        x_i = torch.unsqueeze(x,0).repeat(B, 1, 1)
        x_j = x_i.permute(1,0,2)
        x_ij = torch.cat((x_i, x_j), dim=-1)
        x_ij = torch.unsqueeze(x_ij.permute(2,0,1),0)
        x_ij = self.prediction(x_ij)
        x_ij = torch.squeeze(x_ij, dim=0).permute(1,2,0)
        x_ij = x_ij*attention_repeat
        pred = x_ij.sum(1)/attention_sum
        
        return self.pt(pred)