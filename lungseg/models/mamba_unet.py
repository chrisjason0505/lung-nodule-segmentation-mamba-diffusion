"""3D Mamba-UNet for nodule segmentation and a matched CNN baseline.

Encoder: residual conv stages (local texture/edges, which conv nets learn
efficiently from little data). At the coarse stages each voxel is a token and
tri-directional Mamba layers mix information across the whole cube in linear
time (global context: vessels vs nodule, pleura attachment, ...).

``use_mamba=False`` swaps every Mamba layer for a residual conv block, giving a
plain residual 3D UNet with the same depth and a similar parameter count. That
ablation shows how much the SSM layers themselves contribute.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lungseg.models.ssm import Mamba, build_mamba


def conv_norm_act(cin, cout, k=3, stride=1):
    return nn.Sequential(
        nn.Conv3d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = conv_norm_act(cin, cout, 3, stride)
        self.c2 = nn.Sequential(nn.Conv3d(cout, cout, 3, padding=1, bias=False),
                                nn.InstanceNorm3d(cout, affine=True))
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride, bias=False),
                                   nn.InstanceNorm3d(cout, affine=True)))
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.c2(self.c1(x)) + self.skip(x))


class MambaLayer3D(nn.Module):
    """Pre-norm residual token mixer: 3 scan orders (D-H-W forward, D-H-W backward,
    W-H-D forward), one Mamba each, summed; followed by a channel MLP."""

    ORDERS = ((2, 3, 4), (2, 3, 4), (4, 3, 2))

    def __init__(self, dim, d_state=16, expand=2, mlp_ratio=2.0, use_checkpoint="auto"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mambas = nn.ModuleList([build_mamba(dim, d_state=d_state, expand=expand)
                                     for _ in self.ORDERS])
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.gamma = nn.Parameter(torch.full((dim,), 1e-1))  # small residual scale -> stable start
        if use_checkpoint == "auto":  # fused CUDA kernels don't store the states; the PyTorch scan does
            use_checkpoint = isinstance(self.mambas[0], Mamba)
        self.use_checkpoint = bool(use_checkpoint)

    def _mix(self, x):
        b, c, d, h, w = x.shape
        out = 0
        for i, (order, mamba) in enumerate(zip(self.ORDERS, self.mambas)):
            xp = x.permute(0, *order, 1)                # b, s1, s2, s3, c
            shp = xp.shape
            t = self.norm(xp.reshape(b, -1, c))
            if i == 1:
                t = t.flip(1)
            if self.use_checkpoint and self.training and t.requires_grad:
                # per-direction checkpointing: the PyTorch scan materialises (B, L, D, N) states
                y = checkpoint(mamba, t, use_reentrant=False)
            else:
                y = mamba(t)
            if i == 1:
                y = y.flip(1)
            y = y.reshape(shp)
            inv = [0] * 5
            for dst, src in enumerate((0, *order, 1)):
                inv[src] = dst
            out = out + y.permute(*inv)
        return out

    def _forward(self, x):
        x = x + self.gamma.view(1, -1, 1, 1, 1) * self._mix(x)
        t = x.permute(0, 2, 3, 4, 1)
        t = t + self.mlp(self.norm2(t))
        return t.permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x):
        return self._forward(x)


class MambaUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, widths=(32, 64, 128, 256, 320),
                 mamba_levels=(2, 3, 4), mamba_depth=1, use_mamba=True, d_state=16,
                 deep_supervision=True, use_checkpoint="auto"):
        super().__init__()
        self.use_mamba = use_mamba
        self.mamba_levels = tuple(mamba_levels)
        self.deep_supervision = deep_supervision
        self.stem = nn.Sequential(conv_norm_act(in_channels, widths[0]), ResBlock(widths[0], widths[0]))
        self.down = nn.ModuleList()
        self.mixers = nn.ModuleList()
        for i in range(1, len(widths)):
            self.down.append(ResBlock(widths[i - 1], widths[i], stride=2))
        for i in range(len(widths)):
            if i in self.mamba_levels:
                blocks = [MambaLayer3D(widths[i], d_state=d_state, use_checkpoint=use_checkpoint)
                          if use_mamba else ResBlock(widths[i], widths[i]) for _ in range(mamba_depth)]
                self.mixers.append(nn.Sequential(*blocks))
            else:
                self.mixers.append(nn.Identity())
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(widths[i], widths[i - 1], 2, stride=2))
            self.dec.append(nn.Sequential(ResBlock(2 * widths[i - 1], widths[i - 1]),
                                          ResBlock(widths[i - 1], widths[i - 1])))
        self.head = nn.Conv3d(widths[0], out_channels, 1)
        # deep supervision on the 1/2 and 1/4 resolution decoder outputs
        self.ds_heads = nn.ModuleList([nn.Conv3d(widths[1], out_channels, 1),
                                       nn.Conv3d(widths[2], out_channels, 1)])

    def forward(self, x):
        skips = []
        x = self.mixers[0](self.stem(x))
        skips.append(x)
        for i, down in enumerate(self.down, start=1):
            x = self.mixers[i](down(x))
            skips.append(x)
        x = skips.pop()
        ds = []
        n_up = len(self.up)
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
            level = n_up - 1 - j  # resolution level of x after this step
            if self.training and self.deep_supervision and level in (1, 2):
                ds.append((level, self.ds_heads[level - 1](x)))
        out = self.head(x)
        if self.training and self.deep_supervision:
            return out, [o for _, o in sorted(ds)]
        return out

    def head_parameter_names(self):
        return [n for n, _ in self.named_parameters() if n.startswith(("head.", "ds_heads."))]


def build_model(arch="mamba", out_channels=1, **kw):
    if arch == "mamba":
        return MambaUNet3D(out_channels=out_channels, use_mamba=True, **kw)
    if arch == "unet":
        return MambaUNet3D(out_channels=out_channels, use_mamba=False, **kw)
    raise ValueError(f"unknown arch {arch}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
