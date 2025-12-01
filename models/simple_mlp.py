import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleD3PMMLP(nn.Module):
    """
    Very simple 2-layer MLP for D3PM:
    - Takes x_t (binary MNIST image) and timestep t
    - Outputs logits for x0: [B, K=2, 28, 28]
    - Uses a small timestep embedding
    """

    def __init__(self, img_size=28*28, hidden_dim=1024, K=2, T=100):
        super().__init__()

        self.img_size = img_size
        self.K = K

        # ----- Timestep Embedding -----
        # Learn an embedding for t in {0,...,T-1}
        self.timestep_embed = nn.Embedding(T, hidden_dim)

        # ----- MLP -----
        self.mlp = nn.Sequential(
            nn.Linear(img_size + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, img_size * K),   # Output 2 logits per pixel
        )

    def forward(self, x_t, t):
        """
        x_t: [B, 1, 28, 28] with values {0,1}
        t:   int OR tensor of shape [B]
        """
        B = x_t.shape[0]
        device = x_t.device

        # Flatten input: [B, 784]
        x_flat = x_t.view(B, -1).float()

        # Make timestep a tensor if needed
        if isinstance(t, int):
            t = torch.tensor([t], device=device).long().repeat(B)
        else:
            t = t.to(device).long()

        # Timestep embedding: [B, hidden_dim]
        t_emb = self.timestep_embed(t)

        # Concatenate
        h = torch.cat([x_flat, t_emb], dim=1)

        # MLP forward
        out = self.mlp(h)                    # [B, 784*K]

        # Reshape to logits for each pixel
        out = out.view(B, self.K, 28, 28)    # [B, 2, 28, 28]

        return out
