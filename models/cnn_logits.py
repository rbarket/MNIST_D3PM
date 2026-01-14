# models/cnn_logits.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding (standard in diffusion models).

    Args:
        t: int64 tensor of shape (B,) with values in [1, T]
        dim: embedding dimension
    Returns:
        emb: float tensor of shape (B, dim)
    """
    if t.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"t must be int tensor, got dtype={t.dtype}")
    half = dim // 2
    device = t.device
    t = t.float()

    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device).float() / float(half)
    )  # (half,)
    args = t[:, None] * freqs[None, :]  # (B, half)

    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SimpleTimeMLP(nn.Module):
    """
    Minimal MLP to map sinusoidal embedding -> a vector used to condition conv features.
    """
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


class TimeCondConvBlock(nn.Module):
    """
    A tiny conv block with additive time conditioning:
        h = Conv -> SiLU -> Conv
        h += time_proj(t) (broadcast over H,W)
        h = SiLU(h)
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)

    def forward(self, x: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            t_feat: (B, time_dim)
        """
        h = self.conv1(x)
        h = F.silu(h)
        h = self.conv2(h)

        # Add time conditioning
        t_add = self.time_proj(t_feat).unsqueeze(-1).unsqueeze(-1)  # (B, out_ch, 1, 1)
        h = h + t_add
        h = F.silu(h)
        return h


@dataclass
class SimpleCNNConfig:
    """
    Very small CNN for predicting per-pixel logits for x0 (K classes) from x_t.
    """
    in_channels: int = 1          # MNIST is 1 channel
    num_classes: int = 2          # K=2 for binary pixels
    base_channels: int = 32       # keep small
    time_emb_dim: int = 32        # keep small
    time_hidden_dim: int = 32     # keep small


class SimpleTimeCondCNN(nn.Module):
    """
    Simple CNN that outputs per-pixel logits for p(x0 | xt, t).

    Input:
        xt: (B, 1, 28, 28) with values in {0,1} (long or float)
        t:  (B,) int64 timesteps
    Output:
        logits: (B, K, 28, 28)
    """
    def __init__(self, cfg: SimpleCNNConfig):
        super().__init__()
        self.cfg = cfg

        # Time embedding -> time features used in conv blocks
        self.time_mlp = SimpleTimeMLP(cfg.time_emb_dim, cfg.time_hidden_dim)

        # Very small stack of time-conditioned conv blocks (no downsampling to keep it simple)
        c = cfg.base_channels
        self.in_proj = nn.Conv2d(cfg.in_channels, c, kernel_size=3, padding=1)

        self.block1 = TimeCondConvBlock(c, c, time_dim=cfg.time_hidden_dim)
        self.block2 = TimeCondConvBlock(c, c, time_dim=cfg.time_hidden_dim)
        self.block3 = TimeCondConvBlock(c, c, time_dim=cfg.time_hidden_dim)

        # Output logits per pixel
        self.out_proj = nn.Conv2d(c, cfg.num_classes, kernel_size=1)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Ensure xt is float for convs
        if xt.dtype in (torch.int32, torch.int64):
            xt = xt.float()
        elif not torch.is_floating_point(xt):
            raise TypeError(f"xt must be float or int tensor, got dtype={xt.dtype}")

        if t.ndim != 1:
            raise ValueError(f"t must have shape (B,), got {tuple(t.shape)}")
        if t.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"t must be int tensor, got dtype={t.dtype}")

        # Time embedding
        t_emb = timestep_embedding(t, self.cfg.time_emb_dim)          # (B, time_emb_dim)
        t_feat = self.time_mlp(t_emb)                                  # (B, time_hidden_dim)

        # Conv trunk
        h = self.in_proj(xt)
        h = F.silu(h)
        h = self.block1(h, t_feat)
        h = self.block2(h, t_feat)
        h = self.block3(h, t_feat)

        logits = self.out_proj(h)                                      # (B, K, H, W)
        return logits
