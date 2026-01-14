# training/visualize_forward.py
from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import torch

from data import get_mnist_dataloaders
from diffusion.transition import D3PMForward


@torch.no_grad()
def visualize_forward_process(
    forward: D3PMForward,
    x0: torch.Tensor,          # (B,1,H,W) long in {0..K-1}
    timesteps: List[int],
    *,
    savepath: str | None = None,
    title: str = "",
):
    """
    Visualize x_t sampled from q(x_t | x0) for a single example across timesteps.
    """
    K = forward.K
    x0 = x0[:1]  # take first example, keep batch dim
    device = x0.device

    # sample xt for each timestep
    xt_list = []
    for t in timesteps:
        if t == 0:
            xt_list.append(x0.clone())
        else:
            xt_list.append(forward.sample_xt(x0, t))
    # convert to float images in [0,1]
    imgs = [(xt.float() / float(K - 1)).cpu() for xt in xt_list]

    n = len(timesteps)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.2))
    if n == 1:
        axes = [axes]

    for j, (t, img) in enumerate(zip(timesteps, imgs)):
        axes[j].imshow(img[0, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[j].set_title(f"t={t}")
        axes[j].set_xticks([])
        axes[j].set_yticks([])
        for spine in axes[j].spines.values():
            spine.set_visible(False)

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"Saved: {savepath}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--timesteps",
        type=str,
        default="0,1,5,10,20,50,75,100,150,200,300,500",
        help="comma-separated list of timesteps to visualize; may include 0",
    )
    parser.add_argument("--save", type=str, default="outputs/forward_K4.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load discretized MNIST (returns x0 already in {0..K-1} long)
    train_loader, _ = get_mnist_dataloaders(
        batch_size=args.batch_size,
        num_workers=2,
        K=args.K,
    )
    x0, y = next(iter(train_loader))
    x0 = x0.to(device)

    # Forward process
    if args.schedule == "linear":
        forward = D3PMForward.from_linear_schedule(
            K=args.K,
            T=args.T,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            device=device,
        )
    else:
        forward = D3PMForward.from_cosine_schedule(
            K=args.K,
            T=args.T,
            device=device,
        )

    timesteps = [int(s) for s in args.timesteps.split(",") if s.strip()]
    bad = [t for t in timesteps if t < 0 or t > forward.T]
    if bad:
        raise ValueError(f"Bad timesteps {bad}; must be in [0, {forward.T}]")

    visualize_forward_process(
        forward,
        x0,
        timesteps,
        savepath=args.save,
        title=f"Forward samples q(x_t|x0), K={args.K}, schedule={args.schedule}",
    )


if __name__ == "__main__":
    main()
