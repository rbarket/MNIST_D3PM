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


class ResBlock(nn.Module):
    """
    Residual block with FiLM conditioning (scale/shift) from a conditioning vector.

    x -> GN -> FiLM(cond) -> SiLU -> Conv -> GN -> FiLM(cond) -> SiLU -> Conv -> + skip
    """
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        )

        # Two FiLM projections (one per normalization)
        self.film1 = nn.Linear(cond_dim, 2 * in_ch)   # gamma/beta for norm1 channels
        self.film2 = nn.Linear(cond_dim, 2 * out_ch)  # gamma/beta for norm2 channels

    def _apply_film(self, h: torch.Tensor, film: nn.Linear, cond: torch.Tensor) -> torch.Tensor:
        gb = film(cond)  # (B, 2C)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return h * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self._apply_film(h, self.film1, cond)
        h = self.conv1(F.silu(h))

        h = self.norm2(h)
        h = self._apply_film(h, self.film2, cond)
        h = self.conv2(F.silu(h))

        return h + self.skip(x)


def make_resblock_stack(in_ch: int, out_ch: int, n: int, cond_dim: int) -> nn.ModuleList:
    blocks = [ResBlock(in_ch, out_ch, cond_dim)]
    for _ in range(n - 1):
        blocks.append(ResBlock(out_ch, out_ch, cond_dim))
    return nn.ModuleList(blocks)


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
    num_classes: int = 4      # K (discrete states per pixel)
    base_channels: int = 96   # increased from 64; try 96 first, then 128 if VRAM allows
    time_emb_dim: int = 64
    time_hidden_dim: int = 256
    num_res_blocks: int = 2   # per resolution

    # Conditioning (MNIST labels)
    cond_num_classes: int = 10
    # kept for backward compatibility / config clarity, but stage embeddings use time_emb_dim
    cond_emb_dim: int = 64


class SmallUNetLogits(nn.Module):
    """
    2-level U-Net for predicting logits of x0 given (xt, t, y).

    Input:
      xt: (B,1,H,W) int {0..K-1} or float
      t:  (B,) int timestep
      y:  (B,) int class label (mandatory)
    Output:
      logits: (B,K,H,W)
    """
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.base_channels

        # Stage-specific label embeddings (Option A).
        # Each produces a vector in the same space as the time embedding so we can add: t_emb + y_emb_stage.
        self.label_embs = nn.ModuleDict({
            "down1": nn.Embedding(cfg.cond_num_classes, cfg.time_emb_dim),
            "down2": nn.Embedding(cfg.cond_num_classes, cfg.time_emb_dim),
            "mid":   nn.Embedding(cfg.cond_num_classes, cfg.time_emb_dim),
            "up1":   nn.Embedding(cfg.cond_num_classes, cfg.time_emb_dim),
            "up2":   nn.Embedding(cfg.cond_num_classes, cfg.time_emb_dim),
        })

        # time MLP (produces a shared hidden conditioning space)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_emb_dim, cfg.time_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_hidden_dim, cfg.time_hidden_dim),
        )

        # extra mixing after time_mlp (helps nonlinearly combine time + label)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cfg.time_hidden_dim, cfg.time_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_hidden_dim, cfg.time_hidden_dim),
        )

        # input projection
        self.in_proj = nn.Conv2d(cfg.in_channels, c, kernel_size=3, padding=1)

        # down level (H,W)
        self.down1 = make_resblock_stack(c, c, cfg.num_res_blocks, cfg.time_hidden_dim)
        self.downsample = Downsample(c)

        # down level (H/2, W/2)
        self.down2 = make_resblock_stack(c, 2 * c, cfg.num_res_blocks, cfg.time_hidden_dim)
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

    def forward(self, xt: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if xt.dtype in (torch.int32, torch.int64):
            xt = xt.float()
        elif not torch.is_floating_point(xt):
            raise TypeError(f"xt must be float or int tensor, got {xt.dtype}")

        if t.ndim != 1 or t.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"t must be (B,) int tensor, got shape={tuple(t.shape)} dtype={t.dtype}")

        if y.ndim != 1 or y.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"y must be (B,) int tensor, got shape={tuple(y.shape)} dtype={y.dtype}")

        if y.shape[0] != t.shape[0]:
            raise ValueError(f"batch mismatch: y.shape[0]={y.shape[0]} vs t.shape[0]={t.shape[0]}")

        # Shared time embedding
        t_emb = timestep_embedding(t, self.cfg.time_emb_dim)  # (B, time_emb_dim)

        # Stage-specific conditioning vectors (Option A)
        def stage_cond(stage: str) -> torch.Tensor:
            y_emb = self.label_embs[stage](y)  # (B, time_emb_dim)
            return self.cond_mlp(self.time_mlp(t_emb + y_emb))  # (B, time_hidden_dim)

        t_feat_down1 = stage_cond("down1")
        t_feat_down2 = stage_cond("down2")
        t_feat_mid = stage_cond("mid")
        t_feat_up1 = stage_cond("up1")
        t_feat_up2 = stage_cond("up2")

        # in
        x = self.in_proj(xt)

        # down level 1
        for blk in self.down1:
            x = blk(x, t_feat_down1)
        skip = x  # (B,c,H,W)

        # downsample
        x = self.downsample(x)  # (B,c,H/2,W/2)

        # down level 2
        for blk in self.down2:
            x = blk(x, t_feat_down2)  # (B,2c,H/2,W/2)

        # bottleneck
        x = self.mid1(x, t_feat_mid)
        x = self.mid2(x, t_feat_mid)

        # upsample
        x = self.upsample(x)  # (B,2c,H,W)

        # if sizes mismatch due to odd dimensions, crop to skip
        if x.shape[-2:] != skip.shape[-2:]:
            Hs, Ws = skip.shape[-2], skip.shape[-1]
            x = x[..., :Hs, :Ws]

        # concat skip
        x = torch.cat([x, skip], dim=1)  # (B,2c+c,H,W)

        # up blocks (use separate stage embeddings)
        x = self.up1(x, t_feat_up1)
        x = self.up2(x, t_feat_up2)

        logits = self.out_proj(F.silu(self.out_norm(x)))  # (B,K,H,W)
        return logits
