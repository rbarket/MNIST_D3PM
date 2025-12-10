import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """
    Residual 3x3 conv block with simple timestep conditioning.
    """

    def __init__(self, channels, time_dim=None, groups=8):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, channels) if time_dim is not None else None
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, t_emb=None):
        h = self.conv1(self.norm1(x))
        if self.time_proj is not None and t_emb is not None:
            # inject timestep as channel-wise bias
            h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.conv2(self.norm2(h))
        h = F.silu(h)
        return x + h


class SimpleD3PMCNN(nn.Module):
    """
    Simple CNN for D3PM with the same I/O as the MLP:
      - Input: x_t [B, 1, 28, 28] and timestep t
      - Output: logits for x0 [B, K, 28, 28]
    """

    def __init__(self, base_channels=64, K=2, T=100):
        super().__init__()
        self.K = K
        self.T = T
        time_dim = base_channels

        # timestep embedding
        self.time_embed = nn.Embedding(T, time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # stem + residual stack
        self.stem = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualConvBlock(base_channels, time_dim=time_dim) for _ in range(3)]
        )

        # head to logits
        self.head = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, K, kernel_size=1),
        )

    def forward(self, x_t, t):
        """
        x_t: [B, 1, 28, 28] with values {0,1}
        t:   int OR tensor of shape [B]
        """
        B = x_t.shape[0]
        device = x_t.device

        if isinstance(t, int):
            t = torch.tensor([t], device=device).long().repeat(B)
        else:
            t = t.to(device).long()

        t_emb = self.time_mlp(self.time_embed(t))

        h = self.stem(x_t.float())
        for block in self.blocks:
            h = block(h, t_emb)

        out = self.head(h)  # [B, K, 28, 28]
        return out
