# models/unet_logits.py
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal embedding for integer timesteps.

    Args:
        t: (B,) int tensor
        dim: embedding dimension
    Returns:
        (B, dim) float tensor
    """
    if t.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"t must be int tensor, got {t.dtype}")
    half = dim // 2
    device = t.device
    t = t.float()

    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device).float() / float(half)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
    Choose a GroupNorm group count that divides num_channels.
    """
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return nn.GroupNorm(g, num_channels)

def make_resblock_stack(in_ch: int, out_ch: int, n: int, time_dim: int) -> nn.ModuleList:
    blocks = []
    blocks.append(ResBlock(in_ch, out_ch, time_dim))
    for _ in range(n - 1):
        blocks.append(ResBlock(out_ch, out_ch, time_dim))
    return nn.ModuleList(blocks)


class ResBlock(nn.Module):
    """
    Residual block with additive time conditioning.

    x -> GN+SiLU+Conv -> + time -> GN+SiLU+Conv -> + skip
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_feat).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


@dataclass
class UNetConfig:
    in_channels: int = 1
    num_classes: int = 4      # K
    base_channels: int = 64   # try 64; reduce to 32 if you want lighter
    time_emb_dim: int = 64
    time_hidden_dim: int = 256
    num_res_blocks: int = 2   # per resolution


class SmallUNetLogits(nn.Module):
    """
    2-level U-Net for predicting logits of x0 given (xt, t).

    Input:
      xt: (B,1,H,W) int {0..K-1} or float
      t:  (B,) int timestep
    Output:
      logits: (B,K,H,W)
    """
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.base_channels

        # time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_emb_dim, cfg.time_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_hidden_dim, cfg.time_hidden_dim),
        )

        # input projection
        self.in_proj = nn.Conv2d(cfg.in_channels, c, kernel_size=3, padding=1)

        # down level (H,W)
        self.down1 = make_resblock_stack(c, c, cfg.num_res_blocks, cfg.time_hidden_dim)
        self.downsample = Downsample(c)

        # down level (H/2, W/2)
        self.down2 = make_resblock_stack(c, 2*c, cfg.num_res_blocks, cfg.time_hidden_dim)
        mid_ch = 2 * c

        # bottleneck
        self.mid1 = ResBlock(mid_ch, mid_ch, cfg.time_hidden_dim)
        self.mid2 = ResBlock(mid_ch, mid_ch, cfg.time_hidden_dim)

        # up
        self.upsample = Upsample(mid_ch)

        # after upsample, concat skip from level1: (mid_ch + c) -> c
        self.up1 = ResBlock(mid_ch + c, c, cfg.time_hidden_dim)
        self.up2 = ResBlock(c, c, cfg.time_hidden_dim)

        # output head
        self.out_norm = group_norm(c)
        self.out_proj = nn.Conv2d(c, cfg.num_classes, kernel_size=1)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if xt.dtype in (torch.int32, torch.int64):
            xt = xt.float()
        elif not torch.is_floating_point(xt):
            raise TypeError(f"xt must be float or int tensor, got {xt.dtype}")

        if t.ndim != 1 or t.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"t must be (B,) int tensor, got shape={tuple(t.shape)} dtype={t.dtype}")

        # time features
        t_emb = timestep_embedding(t, self.cfg.time_emb_dim)   # (B, time_emb_dim)
        t_feat = self.time_mlp(t_emb)                          # (B, time_hidden_dim)

        # in
        x = self.in_proj(xt)

        # down level 1
        for blk in self.down1:
            x = blk(x, t_feat)
        skip = x  # (B,c,H,W)

        # downsample
        x = self.downsample(x)  # (B,c,H/2,W/2)

        # down level 2
        for blk in self.down2:
            x = blk(x, t_feat)  # (B,2c,H/2,W/2)

        # bottleneck
        x = self.mid1(x, t_feat)
        x = self.mid2(x, t_feat)

        # upsample
        x = self.upsample(x)  # (B,2c,H,W) (for even sizes)

        # if sizes mismatch due to odd dimensions, crop to skip
        if x.shape[-2:] != skip.shape[-2:]:
            Hs, Ws = skip.shape[-2], skip.shape[-1]
            x = x[..., :Hs, :Ws]

        # concat skip
        x = torch.cat([x, skip], dim=1)  # (B,2c+c,H,W)

        # up blocks
        x = self.up1(x, t_feat)
        x = self.up2(x, t_feat)

        logits = self.out_proj(F.silu(self.out_norm(x)))  # (B,K,H,W)
        return logits
