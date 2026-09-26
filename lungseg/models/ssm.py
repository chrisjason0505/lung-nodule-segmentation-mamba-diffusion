"""Mamba (S6 selective state space) layer with three interchangeable backends.

1. ``mamba_ssm`` fused CUDA kernels (fastest; ``pip install mamba-ssm causal-conv1d``)
2. ``mambapy`` Blelloch parallel scan in pure PyTorch (any GPU / CPU)
3. a sequential reference scan (slow, used for testing)

The pure-PyTorch :class:`Mamba` module has exactly the same parameter names and
shapes as ``mamba_ssm.Mamba``, so checkpoints move freely between backends.
This is a real selective SSM (input-dependent Delta, B, C), not a linear stand-in.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # fused CUDA implementation
    from mamba_ssm import Mamba as _CudaMamba  # type: ignore
    HAS_MAMBA_SSM = True
except Exception:  # pragma: no cover - depends on environment
    _CudaMamba = None
    HAS_MAMBA_SSM = False

try:
    from mambapy.pscan import pscan as _pscan  # type: ignore
    HAS_PSCAN = True
except Exception:  # pragma: no cover
    _pscan = None
    HAS_PSCAN = False


def selective_scan_ref(u, delta, A, B, C, D):
    """Sequential reference. u, delta: (b, d, l); A: (d, n); B, C: (b, n, l); D: (d,)."""
    b, d, l = u.shape
    dA = torch.exp(delta.unsqueeze(-1) * A[:, None, :])                         # b d l n
    dBu = (delta * u).unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)  # b d l n
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for t in range(l):
        h = dA[:, :, t] * h + dBu[:, :, t]
        ys.append((h * C[:, :, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, dim=-1) + u * D.unsqueeze(-1)


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def selective_scan_parallel(u, delta, A, B, C, D):
    """Same maths as :func:`selective_scan_ref` using mambapy's parallel scan."""
    b, d, l = u.shape
    dA = torch.exp(delta.unsqueeze(-1) * A[:, None, :]).transpose(1, 2)                          # b l d n
    dBu = ((delta * u).unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)).transpose(1, 2)  # b l d n
    L2 = _next_pow2(l)
    if L2 != l:  # pad with identity steps (A=1, x=0) at the end: causal, so harmless
        dA = F.pad(dA, (0, 0, 0, 0, 0, L2 - l), value=1.0)
        dBu = F.pad(dBu, (0, 0, 0, 0, 0, L2 - l), value=0.0)
    h = _pscan(dA.contiguous(), dBu.contiguous())[:, :l]                               # b l d n
    y = torch.einsum("bldn,bnl->bdl", h, C)
    return y + u * D.unsqueeze(-1)


class Mamba(nn.Module):
    """Pure PyTorch Mamba block (mirrors ``mamba_ssm.modules.mamba_simple.Mamba``)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=False, conv_bias=True,
                 scan="auto"):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.scan = scan
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # dt init as in the reference implementation
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x):  # x: (b, l, d_model)
        b, l, _ = x.shape
        xz = self.in_proj(x).transpose(1, 2)            # b 2d l
        xs, z = xz.chunk(2, dim=1)
        xs = F.silu(self.conv1d(xs)[..., :l])
        x_dbl = self.x_proj(xs.transpose(1, 2))        # b l (r+2n)
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # the scan is numerically sensitive -> always fp32
        with torch.autocast(device_type=x.device.type, enabled=False):
            ft = torch.float32 if xs.dtype in (torch.float16, torch.bfloat16) else xs.dtype
            delta = F.softplus(F.linear(dt.to(ft), self.dt_proj.weight.to(ft),
                                        self.dt_proj.bias.to(ft))).transpose(1, 2)  # b d l
            A = -torch.exp(self.A_log.to(ft))
            args = (xs.to(ft), delta, A, Bm.to(ft).transpose(1, 2), Cm.to(ft).transpose(1, 2),
                    self.D.to(ft))
            mode = self.scan
            if mode == "auto":
                mode = "parallel" if HAS_PSCAN else "ref"
            y = selective_scan_parallel(*args) if mode == "parallel" else selective_scan_ref(*args)
        y = y.to(z.dtype) * F.silu(z)
        return self.out_proj(y.transpose(1, 2))


def build_mamba(d_model, d_state=16, d_conv=4, expand=2, backend="auto"):
    """``backend``: auto | cuda | torch. ``LUNGSEG_MAMBA_BACKEND`` env var overrides auto."""
    backend = os.environ.get("LUNGSEG_MAMBA_BACKEND", backend) if backend == "auto" else backend
    if backend in ("auto", "cuda") and HAS_MAMBA_SSM and torch.cuda.is_available():
        return _CudaMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    if backend == "cuda":
        raise RuntimeError("mamba_ssm CUDA kernels not available")
    return Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)


def backend_name():
    if HAS_MAMBA_SSM and torch.cuda.is_available() and os.environ.get("LUNGSEG_MAMBA_BACKEND", "auto") != "torch":
        return "mamba_ssm (CUDA)"
    return "pytorch parallel scan (mambapy)" if HAS_PSCAN else "pytorch sequential scan"
