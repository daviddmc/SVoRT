import os
import argparse
import numpy as np
from config import get_config
import torch
from models import *
from data.scan import Scanner
from data.dataset import CombinedDataset


if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="path to the yaml config file", required=True, type=str
    )
    parser.add_argument(
        "--checkpoint", help="path to the checkpoint file", required=True, type=str
    )
    parser.add_argument("--output", help="output folder", required=True, type=str)
    args = parser.parse_args()

    cfg = get_config(args.config)
    # mkdir
    os.makedirs(args.output, exist_ok=True)
    # model
    device = torch.device(cfg["model"]["device"])
    model = globals()[cfg["model"]["model_type"]](**cfg["model"]["model_param"])
    cp = torch.load(args.checkpoint)
    model.to(device)
    model.load_state_dict(cp["model"])
    model.eval()

    # test dataset
    dataset = CombinedDataset(True, cfg["dataset"])
    scanner = Scanner(cfg["scanner"])

    for i in range(len(dataset)):
        # read data
        data = dataset.get_data()
        data = scanner.scan(data)
        for k in data:
            if torch.is_tensor(data[k]):
                data[k] = data[k].to(device, non_blocking=True)
        path_save = os.path.join(args.output, str(i))
        os.makedirs(path_save, exist_ok=True)
        # run models
        transforms = {}
        volumes = {}
        points = {}
        with torch.no_grad():
            transforms, volumes, points = model(data)

        # save transform
        np.save(
            os.path.join(path_save, "transforms.npy"),
            transforms[-1].matrix().detach().cpu().numpy(),
        )
        np.save(
            os.path.join(path_save, "transforms_gt.npy"),
            data["transforms_gt"].detach().cpu().numpy(),
        )
