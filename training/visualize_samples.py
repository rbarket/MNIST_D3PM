# training/visualize_samples.py
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch

from diffusion.transition import D3PMForward
from diffusion.sampling import sample_loop
# from models.cnn_logits import SimpleCNNConfig, SimpleTimeCondCNN
from models.unet_logits import UNetConfig, SmallUNetLogits


## For CNN
# def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
#     cfg = SimpleCNNConfig(
#         in_channels=1,
#         num_classes=2,
#         base_channels=32,
#         time_emb_dim=32,
#         time_hidden_dim=32,
#     )
#     model = SimpleTimeCondCNN(cfg).to(device).eval()

#     state = torch.load(ckpt_path, map_location=device)
#     if isinstance(state, dict) and "state_dict" in state:
#         state = state["state_dict"]
#     model.load_state_dict(state, strict=True)
#     return model

## For UNet
def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    cfg = UNetConfig(
        in_channels=1,
        num_classes=2,
        base_channels=64,      # MUST match what you trained with
        time_emb_dim=64,
        time_hidden_dim=256,
        num_res_blocks=2,
    )
    model = SmallUNetLogits(cfg).to(device).eval()

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model


def resolve_ckpt_path(ckpt: str) -> str:
    if os.path.isabs(ckpt) or os.path.dirname(ckpt):
        return ckpt
    return os.path.join("outputs", "models", ckpt)

@torch.no_grad()
def plot_grid(
    x: torch.Tensor,  # (B,1,H,W) in {0,1}
    *,
    labels: torch.Tensor | None = None,  # (B,) optional
    nrow: int = 8,
    title: str = "",
    savepath: str | None = None,
):
    B = x.shape[0]
    ncol = nrow
    n = min(B, nrow * ncol)

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.5, nrow * 1.8))  # slightly taller
    if nrow == 1:
        axes = axes[None, :]

    x = x[:n].cpu().float()
    if labels is not None:
        labels = labels[:n].detach().cpu()

    for i in range(nrow * ncol):
        r = i // ncol
        c = i % ncol
        ax = axes[r, c]
        ax.axis("off")
        if i < n:
            ax.imshow(x[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
            if labels is not None:
                ax.set_title(f"{int(labels[i].item())}", fontsize=9, pad=2)

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"Saved: {savepath}")

    plt.show()



@torch.no_grad()
def plot_trajectory(
    traj: List[Tuple[int, torch.Tensor]],
    *,
    example_idx: int = 0,
    savepath: str | None = None,
):
    """
    traj: list of (t, x_t_cpu) where x_t_cpu has shape (B,1,H,W)
    Visualize one example across saved timesteps.
    """
    traj = sorted(traj, key=lambda z: z[0], reverse=True)  # t descending

    n = len(traj)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.0, 2.0))

    if n == 1:
        axes = [axes]

    for j, (t, x_t) in enumerate(traj):
        axes[j].imshow(x_t[example_idx, 0].float(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[j].set_title(f"t={t}")
        axes[j].axis("off")

    plt.tight_layout()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"Saved: {savepath}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt_path = resolve_ckpt_path(args.ckpt)
    model = load_model(ckpt_path, device)

    forward = D3PMForward.from_linear_schedule(
        K=2, T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, device=device
    )

    # (A) Final samples grid
    x0 = sample_loop(model, forward, batch_size=args.batch_size, image_size=(28, 28), device=device)
    plot_grid(x0, nrow=8, title="D3PM samples (x0)", savepath="outputs/samples_grid.png")

    # (B) Trajectory snapshots (denoising progression) for a smaller batch
    x0b, traj = sample_loop(
        model,
        forward,
        batch_size=8,
        image_size=(28, 28),
        device=device,
        return_trajectory=True,
        trajectory_steps=[forward.T, forward.T // 2, forward.T // 4, 1, 0],
    )
    plot_trajectory(traj, example_idx=0, savepath="outputs/sampling_trajectory.png")


if __name__ == "__main__":
    main()
