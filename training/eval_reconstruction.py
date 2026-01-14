# training/eval_reconstruction.py
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from diffusion.transition import D3PMForward
from diffusion.posterior import p_theta_xtm1_given_xt
from diffusion.sampling import sample_categorical
from models.unet_logits import UNetConfig, SmallUNetLogits  # or your CNN model


@torch.no_grad()
def reconstruct_from_xt(
    model,
    forward: D3PMForward,
    xt: torch.Tensor,  # (B,1,H,W) long
    t: int,
    *,
    deterministic_last: bool = True,
) -> torch.Tensor:
    """
    Run reverse chain starting from a given x_t down to x0.

    Args:
        xt: starting state at timestep t
        t: integer in [1..T]
    Returns:
        x0_hat: (B,1,H,W) long
    """
    assert xt.ndim == 4 and xt.shape[1] == 1
    device = xt.device
    B = xt.shape[0]

    x = xt
    for step in range(t, 0, -1):
        t_tensor = torch.full((B,), step, device=device, dtype=torch.long)
        logits_x0 = model(x, t_tensor)                       # (B,K,H,W)
        p_xtm1 = p_theta_xtm1_given_xt(forward, logits_x0, x, t_tensor)  # (B,1,H,W,K)

        if deterministic_last and step == 1:
            # argmax over K at final step (crisper reconstructions)
            x = p_xtm1.argmax(dim=-1).to(torch.long)         # (B,1,H,W)
        else:
            x = sample_categorical(p_xtm1)                   # (B,1,H,W)

    return x


@torch.no_grad()
def batch_metrics(x0: torch.Tensor, xhat: torch.Tensor) -> Dict[str, float]:
    """
    x0, xhat: (B,1,H,W) long in {0,1}
    """
    x0f = x0.float()
    xhf = xhat.float()
    eq = (x0 == xhat)

    pixel_acc = eq.float().mean().item()

    # foreground/background accuracies (binary)
    fg_mask = (x0 == 1)
    bg_mask = (x0 == 0)
    fg_acc = (eq[fg_mask].float().mean().item()) if fg_mask.any() else 0.0
    bg_acc = (eq[bg_mask].float().mean().item()) if bg_mask.any() else 0.0
    bal_acc = 0.5 * (fg_acc + bg_acc)

    # Hamming = 1 - pixel_acc for binary (same as mean absolute error here)
    hamming = (1.0 - pixel_acc)
    mae = (x0f - xhf).abs().mean().item()

    return {
        "pixel_acc": pixel_acc,
        "fg_acc": fg_acc,
        "bg_acc": bg_acc,
        "balanced_acc": bal_acc,
        "hamming": hamming,
        "mae": mae,
    }


def load_model(ckpt: str, device: torch.device):
    cfg = UNetConfig(
        in_channels=1,
        num_classes=2,
        base_channels=64,
        time_emb_dim=64,
        time_hidden_dim=256,
        num_res_blocks=2,
    )
    model = SmallUNetLogits(cfg).to(device).eval()

    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model


def resolve_ckpt_path(ckpt: str) -> str:
    if os.path.isabs(ckpt) or os.path.dirname(ckpt):
        return ckpt
    return os.path.join("outputs", "models", ckpt)


def make_test_loader(batch_size: int, num_workers: int = 4):
    ds = MNIST(
        "./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def binarize(x: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """
    x: (B,1,H,W) float in [0,1]
    returns: (B,1,H,W) long in {0,1}
    """
    return (x >= thresh).to(torch.long)


@torch.no_grad()
def eval_reconstruction(
    model,
    forward: D3PMForward,
    loader: DataLoader,
    timesteps: List[int],
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> Dict[int, Dict[str, float]]:
    """
    For each timestep t: x_t ~ q(x_t|x0), reconstruct x0_hat, compute metrics averaged over test set.
    """
    agg = {t: {"n": 0, "pixel_acc": 0.0, "fg_acc": 0.0, "bg_acc": 0.0, "balanced_acc": 0.0, "hamming": 0.0, "mae": 0.0}
           for t in timesteps}

    for b_idx, (x, y) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        x = x.to(device)
        x0 = binarize(x)  # (B,1,H,W) long

        for t in timesteps:
            xt = forward.sample_xt(x0, t)  # (B,1,H,W) long
            xhat = reconstruct_from_xt(model, forward, xt, t, deterministic_last=True)
            m = batch_metrics(x0, xhat)

            n = x0.shape[0]
            agg[t]["n"] += n
            for k in m:
                agg[t][k] += m[k] * n

    # normalize
    out = {}
    for t in timesteps:
        n = agg[t]["n"]
        out[t] = {k: (agg[t][k] / max(1, n)) for k in agg[t] if k != "n"}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=0, help="0 means all test batches")
    p.add_argument("--timesteps", type=str, default="1,20,100,200")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt_path = resolve_ckpt_path(args.ckpt)
    model = load_model(ckpt_path, device)
    forward = D3PMForward.from_linear_schedule(
        K=2, T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, device=device
    )

    loader = make_test_loader(args.batch_size, args.num_workers)
    timesteps = [int(s) for s in args.timesteps.split(",") if s.strip()]
    max_batches = None if args.max_batches == 0 else args.max_batches

    results = eval_reconstruction(model, forward, loader, timesteps, device, max_batches=max_batches)

    print("\nReconstruction metrics (test set):")
    for t in timesteps:
        r = results[t]
        print(
            f"t={t:4d}  "
            f"pixel_acc={r['pixel_acc']:.4f}  "
            f"bal_acc={r['balanced_acc']:.4f}  "
            f"fg_acc={r['fg_acc']:.4f}  "
            f"bg_acc={r['bg_acc']:.4f}  "
            f"hamming={r['hamming']:.4f}  "
            f"mae={r['mae']:.4f}"
        )


if __name__ == "__main__":
    main()
