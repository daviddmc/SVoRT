import torch
import torch.nn as nn
import torch.nn.functional as F
from transform import mat_update_resolution, point2mat
from .attention import TransformerEncoder, PositionalEncoding, ResNet
from slice_acquisition import slice_acquisition
from .reconstruction import SRR

# from .vit import SimpleViT


class SRRtransformer(nn.Module):
    def __init__(
        self,
        n_res=34,
        n_layers=4,
        n_head=4,
        d_in=8,
        d_out=1,
        d_model=256,
        d_inner=512,
        dropout=0.1,
    ):
        super().__init__()
        self.srr = SRR(n_iter=2, use_CG=True)
        self.img_encoder = ResNet(
            n_res=n_res, d_model=d_model, pretrained=False, d_in=2
        )
        self.pos_emb = PositionalEncoding(d_model, d_in)
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
        )
        self.fc = nn.Linear(d_model, d_out)

    def forward(
        self, theta, transforms, slices, volume, params, idx, attn_mask, feat_stack=None
    ):
        slices_est = slice_acquisition(
            transforms,
            volume,
            None,
            None,
            params["psf"],
            params["slice_shape"],
            params["res_s"] / params["res_r"],
            False,
            params["interp_psf"],
        )
        idx = torch.cat((theta, idx), -1)
        x = torch.cat((slices, slices_est), 1)
        pe = self.pos_emb(idx)
        x = self.img_encoder(x)
        if feat_stack is not None:
            feat_stack = self.fc_s(feat_stack)
            x = torch.cat([feat_stack, x], 0)
        x, _ = self.encoder(x, pe, attn_mask)
        if feat_stack is not None:
            feat_stack = x[: feat_stack.shape[0]]
            x = x[feat_stack.shape[0] :]
        x = self.fc(x)
        x = F.softmax(x, dim=0) * x.shape[0]
        x = torch.clamp(x, max=3.0)
        volume = self.srr(transforms, slices, volume, params, x.view(-1, 1, 1, 1))
        return volume, x


class SVRtransformer(nn.Module):
    def __init__(
        self,
        n_res=34,
        n_layers=4,
        n_head=4,
        d_in=8,
        d_out=6,
        d_model=256,
        d_inner=512,
        dropout=0.1,
        res_d_in=3,
        res_scale=1,
    ):
        super().__init__()
        if isinstance(n_res, nn.Module):
            self.img_encoder = n_res
        else:
            self.img_encoder = ResNet(
                n_res=n_res, d_model=d_model, pretrained=False, d_in=res_d_in
            )
        self.pos_emb = PositionalEncoding(d_model, d_in)
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
        )
        self.fc = nn.Linear(d_model, d_out)
        self.res_scale = res_scale
        self.res_d_in = res_d_in

    def pos_augment(self, slices, slices_est):
        n, _, h, w = slices.shape
        y = torch.linspace(-(h - 1) / 256, (h - 1) / 256, h, device=slices.device)
        x = torch.linspace(-(w - 1) / 256, (w - 1) / 256, w, device=slices.device)
        y, x = torch.meshgrid(y, x, indexing="ij")  # hxw
        if slices_est is not None:
            slices = torch.cat(
                [
                    slices,
                    slices_est,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        else:
            if self.res_d_in == 3:
                slices = torch.cat(
                    [
                        slices,
                        y.view(1, 1, h, w).expand(n, -1, -1, -1),
                        x.view(1, 1, h, w).expand(n, -1, -1, -1),
                    ],
                    1,
                )
            else:
                slices = torch.cat(
                    [
                        slices,
                        0 * slices,
                        y.view(1, 1, h, w).expand(n, -1, -1, -1),
                        x.view(1, 1, h, w).expand(n, -1, -1, -1),
                    ],
                    1,
                )
        return slices

    def forward(self, theta, slices, pos, volume, params, attn_mask, feat_stack=None):
        y = volume
        if y is not None:
            with torch.no_grad():
                transforms = mat_update_resolution(point2mat(theta), 1, params["res_r"])
                y = slice_acquisition(
                    transforms,
                    y,
                    None,
                    None,
                    params["psf"],
                    params["slice_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    params["interp_psf"],
                )
        pos = torch.cat((theta, pos), -1)
        pe = self.pos_emb(pos)
        slices = self.pos_augment(slices, y)
        x = self.img_encoder(slices)
        if feat_stack is not None:
            x = torch.cat([feat_stack, x], 0)
        x, attn = self.encoder(x, pe, attn_mask)
        if feat_stack is not None:
            feat_stack = x[: feat_stack.shape[0]]
            x = x[feat_stack.shape[0] :]
        x = self.fc(x)
        return theta + x * self.res_scale, attn


class SVRtransformerV2(nn.Module):
    def __init__(
        self,
        n_res=50,
        n_layers=4,
        n_head=4,
        d_in=8,
        d_out=6,
        d_model=256,
        d_inner=512,
        dropout=0.1,
        n_channels=2,
    ):
        super().__init__()
        # self.img_encoder = ViT(channels=n_channels, num_classes=d_model)
        self.img_encoder = ResNet(
            n_res=n_res, d_model=d_model, pretrained=False, d_in=n_channels + 2
        )
        self.pos_emb = PositionalEncoding(d_model, d_in)
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            activation_attn="softmax",
            activation_ff="gelu",
            prenorm=False,
        )
        self.fc = nn.Linear(d_model, d_out)
        self.fc_score = nn.Linear(d_model, 1)

    def pos_augment(self, slices, slices_est):
        n, _, h, w = slices.shape
        y = torch.linspace(-(h - 1) / 256, (h - 1) / 256, h, device=slices.device)
        x = torch.linspace(-(w - 1) / 256, (w - 1) / 256, w, device=slices.device)
        y, x = torch.meshgrid(y, x, indexing="ij")  # hxw
        if slices_est is not None:
            slices = torch.cat(
                [
                    slices,
                    slices_est,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        else:
            slices = torch.cat(
                [
                    slices,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        return slices

    def forward(self, theta, slices, pos, volume, params, attn_mask):
        y = volume
        if y is not None:
            with torch.no_grad():
                transforms = mat_update_resolution(point2mat(theta), 1, params["res_r"])
                y = slice_acquisition(
                    transforms,
                    y,
                    None,
                    None,
                    params["psf"],
                    params["slice_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    params["interp_psf"],
                )
        pos = torch.cat((theta, pos), -1)
        pe = self.pos_emb(pos)
        if isinstance(self.img_encoder, ResNet):
            slices = self.pos_augment(slices, y)
        else:  # ViT
            if y is not None:
                slices = torch.cat([slices, y], 1)
        x = self.img_encoder(slices)
        x, attn = self.encoder(x, pe, attn_mask)
        dtheta = self.fc(x)

        score = self.fc_score(x)
        score = F.softmax(score, dim=0) * score.shape[0]
        score = torch.clamp(score, max=3.0)

        return theta + dtheta, score, attn
