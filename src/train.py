import os
from config import get_config
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import argparse
from models import *
from data.io import save_volume
from data.scan import Scanner
from data.dataset import CombinedDataset
import time
import datetime
from transform import RigidTransform, mat2point


def read_data(cfg_dataset, cfg_scanner, queue):
    import traceback

    dataset = CombinedDataset(False, cfg_dataset)
    scanner = Scanner(cfg_scanner)
    try:
        while True:
            data = dataset.get_data()
            data = scanner.scan(data)
            for k in data:
                if torch.is_tensor(data[k]):
                    data[k] = data[k].cpu()
            queue.put(data)
    except Exception as e:
        print("read_data failed!")
        traceback.print_exc()


class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.t_total - step)
            / float(max(1.0, self.t_total - self.warmup_steps)),
        )


class MovingAverage:
    def __init__(self, alpha):
        assert 0 <= alpha < 1
        self.alpha = alpha
        self._value = dict()

    def to_dict(self):
        return {"alpha": self.alpha, "value": self._value}

    def from_dict(self, d):
        self.alpha = d["alpha"]
        self._value = d["value"]

    def __call__(self, key, value):
        if key not in self._value:
            self._value[key] = (0, 0)
        num, v = self._value[key]
        num += 1
        if self.alpha:
            v = v * self.alpha + value * (1 - self.alpha)
        else:
            v += value
        self._value[key] = (num, v)

    def __str__(self):
        s = ""
        for key in self._value:
            num, v = self._value[key]
            if self.alpha:
                s += "%s = %.3e  " % (key, v / (1 - self.alpha**num))
            else:
                s += "%s = %.3e  " % (key, v / num)
        if len(self._value) > 0:
            return ("iter = %d  " % num) + s
        else:
            return s

    def header(self):
        return "iter," + ",".join(self._value.keys())

    def value(self):
        values = []
        for key in self._value:
            num, v = self._value[key]
            if self.alpha:
                values.append(v / (1 - self.alpha**num))
            else:
                values.append(v / num)
        if len(self._value) > 0:
            return [num] + values
        else:
            return values


def img_loss(volumes, volume_gt, beta=0.01):
    loss_img = [
        F.smooth_l1_loss(v, volume_gt, beta=beta, reduction="mean") for v in volumes
    ]
    return loss_img


def trans_loss(transforms, transforms_gt):
    transfroms_err = [transforms_gt.inv().compose(t) for t in transforms]
    err = [t_err.axisangle() for t_err in transfroms_err]
    loss_R = [torch.mean(e[:, :3] ** 2) for e in err]
    loss_T = [torch.mean(e[:, 3:] ** 2) for e in err]
    return loss_R, loss_T


def point_loss(ps, transforms_gt, sx, sy, rs):
    p_gt = mat2point(transforms_gt, sx, sy, rs)
    loss_p = [F.mse_loss(p, p_gt) for p in ps]
    return loss_p


if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="path to the yaml config file", required=True, type=str
    )
    parser.add_argument("--output", help="output folder", required=True, type=str)
    args = parser.parse_args()

    cfg = get_config(args.config)
    # mkdir
    os.makedirs(os.path.join(args.output, "outputs"), exist_ok=True)
    # model and optimizer
    device = torch.device(cfg["model"]["device"])
    model = globals()[cfg["model"]["model_type"]](**cfg["model"]["model_param"]).to(
        device
    )
    n_train = cfg["model"]["n_train"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["model"]["lr"],
        weight_decay=cfg["model"]["weight_decay"],
    )
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=cfg["model"]["warmup_steps"],
        t_total=n_train / cfg["model"]["batch_size"],
    )
    average = MovingAverage(0.999)
    # read data with multiprocessing
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(4)
    p = ctx.Process(target=read_data, args=(cfg["dataset"], cfg["scanner"], queue))
    p.daemon = True
    p.start()
    # main loop
    t_start = time.time()
    for i in range(n_train):
        # read data
        data = queue.get()
        for k in data:
            if torch.is_tensor(data[k]):
                data[k] = data[k].to(device, non_blocking=True)
        # forward and backward
        try:
            transforms, volumes, points = model(data)
            loss_p = point_loss(
                points,
                data["transforms_gt"],
                data["slice_shape"][0],
                data["slice_shape"][1],
                data["resolution_slice"],
            )
            loss_R, loss_T = trans_loss(
                transforms, RigidTransform(data["transforms_gt"])
            )
            loss_img = img_loss(volumes, data["volume_gt"])

            loss = (
                cfg["model"].get("weight_point", 0) * sum(loss_p)
                + cfg["model"].get("weight_img", 0) * sum(loss_img)
                + cfg["model"].get("weight_T", 0) * sum(loss_T)
                + cfg["model"].get("weight_R", 0) * sum(loss_R)
            )

            (loss / cfg["model"]["batch_size"]).backward()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM")
                del data, loss, loss_R, loss_T, loss_p, loss_img, transforms, volumes
                for _p in model.parameters():
                    if _p.grad is not None:
                        del _p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise
        # stats
        if np.isfinite(
            [
                loss_R[-1].item(),
                loss_T[-1].item(),
                loss_img[-1].item(),
                loss_p[-1].item(),
            ]
        ).all():
            average("rot", loss_R[-1].item())
            average("tran", loss_T[-1].item())
            average("img", loss_img[-1].item())
            average("point", loss_p[-1].item())
        else:
            print("warning: loss is nan or inf")
            print(
                [
                    loss_R[-1].item(),
                    loss_T[-1].item(),
                    loss_img[-1].item(),
                    loss_p[-1].item(),
                ]
            )
        # optimizer step
        if (i + 1) % cfg["model"]["batch_size"] == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 500.0).item()
            if np.isfinite(grad_norm):
                optimizer.step()
                scheduler.step()
            else:
                print("warning: grad norm is nan or inf")
                print(grad_norm)
            optimizer.zero_grad()
        # print out
        if (i + 1) % 100 == 0:
            volume_gt = data["volume_gt"]
            fname = os.path.join(args.output, "outputs")
            save_volume(
                os.path.join(fname, "gt.nii.gz"), volume_gt, data["resolution_recon"]
            )
            save_volume(
                os.path.join(fname, "out.nii.gz"), volumes[-1], data["resolution_recon"]
            )
            fname = os.path.join(args.output, "loss.csv")
            header = "" if os.path.exists(fname) else average.header()
            with open(fname, "ab") as f:
                np.savetxt(
                    f,
                    np.array(average.value())[None],
                    delimiter=",",
                    header=header,
                    comments="",
                )
            print(
                str(datetime.timedelta(seconds=int(time.time() - t_start)))
                + "  "
                + str(average)
                + "  lr = %.3e" % optimizer.param_groups[0]["lr"]
            )
        # checkpointing
        if (i + 1) % 1000 == 0:
            torch.save(
                {"iter": i, "model": model.state_dict(), "loss": average.to_dict()},
                os.path.join(args.output, "checkpoint.pt"),
            )
        i += 1
