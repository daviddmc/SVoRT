import os
import sys
import numpy as np
from config import get_config
import torch
from models import SVoRT
from data.scan import Scanner
from data.dataset import RegisteredFetaDataset


if __name__ == '__main__':
    assert len(sys.argv) == 4
    # parameters
    name = sys.argv[1]
    path_cp = sys.argv[2]
    path_out = sys.argv[3]
    
    cfg = get_config('config_' + name)
    # mkdir
    os.makedirs(path_out, exist_ok=True)
    # model
    device = torch.device(cfg['model']['device'])
    model = globals()[cfg['model']['model_type']](**cfg['model']['model_param'])
    cp = os.path.join(path_cp)
    cp = torch.load(cp)
    model.to(device)
    model.load_state_dict(cp['model'])
    model.eval()
    
    # test dataset
    dataset = RegisteredFetaDataset(True, cfg['dataset'])
    scanner = Scanner(cfg['scanner'])

    for i in range(len(dataset)):
        # read data
        data = dataset.get_data()
        data = scanner.scan(data)
        for k in data:
            if torch.is_tensor(data[k]):
                data[k] = data[k].to(device, non_blocking=True)
        path_save = os.path.join(path_out, str(i))
        os.makedirs(path_save, exist_ok=True)
        # run models
        transforms = {}
        volumes = {}
        points = {}
        with torch.no_grad():
            transforms, volumes, points = model(data)
                
        # save transform
        np.save(os.path.join(path_save, 'transforms.npy'), transforms[-1].matrix().detach().cpu().numpy())
        np.save(os.path.join(path_save, 'transforms_gt.npy'), data['transforms_gt'].detach().cpu().numpy())
