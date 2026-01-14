# training/plot_reconstruction_metrics.py
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from diffusion.transition import D3PMForward
from diffusion.posterior import p_theta_xtm1_given_xt
from diffusion.sampling import sample_categorical

from models.unet_logits import UNetConfig, SmallUNetLogits  # swap if needed


def discretize(x: torch.Tensor, K: int) -> torch.Tensor:
    return (x * (K - 1)).round().long().clamp(0, K - 1)


def load_model(ckpt: str, device: torch.device, K: int) -> torch.nn.Module:
    cfg = UNetConfig(
        in_channels=1,
        num_classes=K,
        base_channels=64,      # MUST match training
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


@torch.no_grad()
def reconstruct_from_xt(
    model: torch.nn.Module,
    forward: D3PMForward,
    xt: torch.Tensor,
    t_start: int,
    *,
    deterministic_last: bool = True,
) -> torch.Tensor:
    """
    Reverse chain from given x_{t_start} to x_0.
    """
    device = xt.device
    B = xt.shape[0]
    x = xt

    for step in range(t_start, 0, -1):
        t = torch.full((B,), step, device=device, dtype=torch.long)
        logits_x0 = model(x, t)  # (B,K,H,W)
        p_xtm1 = p_theta_xtm1_given_xt(forward, logits_x0, x, t)  # (B,1,H,W,K)

        if deterministic_last and step == 1:
            x = p_xtm1.argmax(dim=-1).to(torch.long)
        else:
            x = sample_categorical(p_xtm1)

    return x


@torch.no_grad()
def batch_metrics(x0: torch.Tensor, xhat: torch.Tensor, K: int) -> Dict[str, float]:
    eq = (x0 == xhat)
    pixel_acc = eq.float().mean().item()
    hamming = 1.0 - pixel_acc

    # macro acc across present classes
    x0f = x0.view(-1)
    xhf = xhat.view(-1)
    accs = []
    for k in range(K):
        m = (x0f == k)
        if m.any():
            accs.append((xhf[m] == k).float().mean().item())
    macro_acc = float(sum(accs) / len(accs)) if accs else 0.0

    # MAE in normalized intensity space (so comparable across K)
    mae = (x0.float() - xhat.float()).abs().mean().item() / max(1.0, float(K - 1))

    out = {
        "pixel_acc": pixel_acc,
        "macro_acc": macro_acc,
        "hamming": hamming,
        "mae": mae,
    }

    # Keep your old binary metrics for K=2 only
    if K == 2:
        fg = (x0 == 1)
        bg = (x0 == 0)
        fg_acc = (eq[fg].float().mean().item()) if fg.any() else 0.0
        bg_acc = (eq[bg].float().mean().item()) if bg.any() else 0.0
        balanced_acc = 0.5 * (fg_acc + bg_acc)
        out.update({"balanced_acc": balanced_acc, "fg_acc": fg_acc, "bg_acc": bg_acc})

    return out



def make_test_loader(batch_size: int, num_workers: int) -> DataLoader:
    ds = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )


@torch.no_grad()
def eval_per_timestep(
    model: torch.nn.Module,
    forward: D3PMForward,
    loader: DataLoader,
    timesteps: List[int],
    device: torch.device,
    *,
    batches_per_t: int,
    deterministic_last: bool,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate each timestep t on only `batches_per_t` batches (fast).
    """
    results: Dict[int, Dict[str, float]] = {}

    for t in timesteps:
        n_total = 0
        sums = {k: 0.0 for k in ["pixel_acc", "balanced_acc", "fg_acc", "bg_acc", "hamming", "mae"]}

        it = iter(loader)
        for _ in range(batches_per_t):
            x, _ = next(it)
            x = x.to(device)
            x0 = discretize(x, forward.K)

            xt = forward.sample_xt(x0, t)
            xhat = reconstruct_from_xt(model, forward, xt, t, deterministic_last=deterministic_last)
            m = batch_metrics(x0, xhat, forward.K)

            bsz = x0.shape[0]
            n_total += bsz
            for k in sums:
                sums[k] += m[k] * bsz

        results[t] = {k: sums[k] / max(1, n_total) for k in sums}

        print(
            f"t={t:4d}  "
            f"pixel_acc={results[t]['pixel_acc']:.4f}  "
            f"bal_acc={results[t]['balanced_acc']:.4f}  "
            f"fg_acc={results[t]['fg_acc']:.4f}  "
            f"bg_acc={results[t]['bg_acc']:.4f}"
        )

    return results


def save_csv(results: Dict[int, Dict[str, float]], outpath: str):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    timesteps_sorted = sorted(results.keys())
    fieldnames = ["t"] + list(next(iter(results.values())).keys())

    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in timesteps_sorted:
            row = {"t": t, **results[t]}
            w.writerow(row)

    print("Saved:", outpath)


def plot_metric(timesteps: List[int], values: List[float], metric: str, outdir: str):
    plt.figure()
    plt.plot(timesteps, values, marker="o")
    plt.xlabel("t (noise level)")
    plt.ylabel(metric)
    plt.title(f"Reconstruction {metric} vs t")
    plt.grid(True, alpha=0.3)
    path = os.path.join(outdir, f"{metric}_vs_t.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batches_per_t", type=int, default=2, help="how many test batches to average per timestep")
    p.add_argument("--deterministic_last", action="store_true", help="use argmax at final step t=1")
    p.add_argument(
        "--timesteps",
        type=str,
        default="1,5,10,20,30,40,50,60,75,100,125,150,200,250,300,400,500",
    )
    
    p.add_argument("--K", type=int, default=2)
    p.add_argument(
        "--out_subdir",
        type=str,
        default="",
        help="optional subfolder name under outputs/metrics/ to save plots+csv",
    )

    args = p.parse_args()
    
    base_outdir = os.path.join("outputs", "metrics")
    outdir = base_outdir if not args.out_subdir else os.path.join(base_outdir, args.out_subdir)
    os.makedirs(outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    timesteps = [int(s) for s in args.timesteps.split(",") if s.strip()]
    bad = [t for t in timesteps if t < 1 or t > args.T]
    if bad:
        raise ValueError(f"Bad timesteps {bad}; must be within [1, {args.T}].")

    ckpt_path = resolve_ckpt_path(args.ckpt)
    model = load_model(ckpt_path, device, args.K)
    forward = D3PMForward.from_linear_schedule(
        K=args.K, T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, device=device
    )
    print("forward.T =", forward.T)

    loader = make_test_loader(args.batch_size, args.num_workers)

    results = eval_per_timestep(
        model,
        forward,
        loader,
        timesteps,
        device,
        batches_per_t=args.batches_per_t,
        deterministic_last=args.deterministic_last,
    )

    # Save CSV
    csv_path = os.path.join(outdir, "reconstruction_metrics_fast.csv")
    save_csv(results, csv_path)

    # Save plots
    timesteps_sorted = sorted(results.keys())
    for metric in ["pixel_acc", "balanced_acc", "fg_acc", "bg_acc", "hamming", "mae"]:
        vals = [results[t][metric] for t in timesteps_sorted]
        plot_metric(timesteps_sorted, vals, metric, outdir)

    # Combined accuracy plot
    plt.figure()
    plt.plot(timesteps_sorted, [results[t]["pixel_acc"] for t in timesteps_sorted], marker="o", label="pixel_acc")
    plt.plot(timesteps_sorted, [results[t]["balanced_acc"] for t in timesteps_sorted], marker="o", label="balanced_acc")
    plt.plot(timesteps_sorted, [results[t]["fg_acc"] for t in timesteps_sorted], marker="o", label="fg_acc")
    plt.plot(timesteps_sorted, [results[t]["bg_acc"] for t in timesteps_sorted], marker="o", label="bg_acc")
    plt.xlabel("t (noise level)")
    plt.ylabel("accuracy")
    plt.title("Reconstruction accuracies vs t (fast)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    combined_path = os.path.join(outdir, "accuracies_vs_t_fast.png")
    plt.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", combined_path)


if __name__ == "__main__":
    main()
