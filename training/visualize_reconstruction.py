# training/visualize_reconstruction.py
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from diffusion.transition import D3PMForward
from diffusion.posterior import p_theta_xtm1_given_xt
from diffusion.sampling import sample_categorical
from models.unet_logits import UNetConfig, SmallUNetLogits  # swap to CNN if needed


def binarize(x: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """x: (B,1,H,W) float in [0,1] -> (B,1,H,W) long in {0,1}"""
    return (x >= thresh).to(torch.long)


def load_model(ckpt: str, device: torch.device, K: int) -> torch.nn.Module:
    cfg = UNetConfig(
        in_channels=1,
        num_classes=K,
        base_channels=64,      # must match your training
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
    if os.path.isabs(ckpt):
        return ckpt
    if os.path.exists(ckpt):
        return ckpt
    candidates = [
        os.path.join("models", ckpt),
        os.path.join("outputs", "model", ckpt),
        os.path.join("outputs", "models", ckpt),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


@torch.no_grad()
def reconstruct_with_snapshots(
    model: torch.nn.Module,
    forward: D3PMForward,
    xt: torch.Tensor,          # (B,1,H,W) long
    t_start: int,
    *,
    y: torch.Tensor,
    snapshot_ts: list[int],    # timesteps to record (include t_start and 0 if desired)
    deterministic_last: bool = True,
):
    """
    Run reverse chain from x_{t_start} to x_0, recording x_t at selected timesteps.

    Returns:
        snapshots: dict {timestep:int -> x_t tensor (B,1,H,W) long on CPU}
    """
    device = xt.device
    B = xt.shape[0]
    x = xt

    snapshot_set = set(snapshot_ts)
    snapshots = {}
    if t_start in snapshot_set:
        snapshots[t_start] = x.detach().cpu()

    for step in range(t_start, 0, -1):
        t = torch.full((B,), step, device=device, dtype=torch.long)
        logits_x0 = model(x, t, y)  # (B,K,H,W)
        p_xtm1 = p_theta_xtm1_given_xt(forward, logits_x0, x, t)  # (B,1,H,W,K)

        if deterministic_last and step == 1:
            x = p_xtm1.argmax(dim=-1).to(torch.long)
        else:
            x = sample_categorical(p_xtm1)

        # after updating, x is x_{step-1}
        if (step - 1) in snapshot_set:
            snapshots[step - 1] = x.detach().cpu()

    return snapshots


def show_recon_snapshots(
    x0_true: torch.Tensor,
    snapshots: dict[int, torch.Tensor],
    labels: torch.Tensor,
    *,
    order: list[int],
    K: int,
    savepath: str | None = None,
):
    x0_true = x0_true.detach().cpu().float() / (K - 1)
    labels = labels.detach().cpu()

    # snapshots: {t -> (B,1,H,W)}
    B = x0_true.shape[0]
    C = 1 + len(order)   # +1 for true x0

    fig, axes = plt.subplots(B, C, figsize=(C * 2.0, B * 2.4))
    if B == 1:
        axes = axes[None, :]

    def hide_axis_keep_labels(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for i in range(B):
        # ---- column 0: true x0 ----
        ax = axes[i, 0]
        ax.imshow(x0_true[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
        hide_axis_keep_labels(ax)
        if i == 0:
            ax.set_title("true x0")
        ax.set_xlabel(f"y={int(labels[i].item())}", fontsize=10, labelpad=8)

        # ---- remaining columns: snapshots ----
        for j, t in enumerate(order):
            ax = axes[i, j + 1]
            x = snapshots[t][i, 0].float() / (K - 1)
            ax.imshow(x, cmap="gray", vmin=0.0, vmax=1.0)
            hide_axis_keep_labels(ax)
            if i == 0:
                ax.set_title(f"t={t}")

    plt.tight_layout()
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"Saved: {savepath}")
    plt.show()




def show_recon_triplets(x0: torch.Tensor, xt: torch.Tensor, xhat: torch.Tensor, labels: torch.Tensor, t: int,
                        savepath: str | None = None):
    """
    Displays rows: [x0 | xt | xhat] for each example.
    """
    x0 = x0.detach().cpu().float()
    xt = xt.detach().cpu().float()
    xhat = xhat.detach().cpu().float()
    labels = labels.detach().cpu()

    B = x0.shape[0]
    fig, axes = plt.subplots(B, 3, figsize=(6.5, 2.2 * B))

    if B == 1:
        axes = axes[None, :]

    for i in range(B):
        axes[i, 0].imshow(x0[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 0].set_title(f"x0 (label={int(labels[i])})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(xt[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title(f"x_t (t={t})")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(xhat[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title("recon x̂0")
        axes[i, 2].axis("off")

    plt.tight_layout()
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"Saved: {savepath}")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.2)
    p.add_argument("--t", type=int, default=250, help="reconstruction start timestep")
    p.add_argument("--n", type=int, default=4, help="number of examples to visualize")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, default="reconstruction_snapshots.png")
    p.add_argument("--K", type=int, default=4)
    p.add_argument(
    "--schedule",
    type=str,
    default="cosine",
    choices=["linear", "cosine"],
 )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt_path = resolve_ckpt_path(args.ckpt)
    model = load_model(ckpt_path, device, args.K)

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



    # Load MNIST test set
    ds = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=args.n, shuffle=True, num_workers=2, pin_memory=True)
    x, y = next(iter(loader))

    x = x.to(device)
    y = y.to(device)

    x0 = binarize(x)  # (B,1,28,28) long {0,1}
    xt = forward.sample_xt(x0, args.t)  # (B,1,28,28) long

        # Choose which timesteps to visualize (must be within [0, args.t])
    snapshot_ts = [args.t, args.t // 2, args.t // 5, 50, 20, 10, 1, 0]
    snapshot_ts = [s for s in snapshot_ts if 0 <= s <= args.t]
    snapshot_ts = sorted(set(snapshot_ts), reverse=True)

    snapshots = reconstruct_with_snapshots(
        model,
        forward,
        xt,
        args.t,
        snapshot_ts=snapshot_ts,
        y=y,
        deterministic_last=True,
    )

    name = args.name
    if not name.lower().endswith(".png"):
        name = f"{name}.png"
    savepath = os.path.join("outputs/reconstruction_snapshots", name)

    show_recon_snapshots(
        x0,
        snapshots,
        y,
        order=snapshot_ts,
        K=args.K,
        savepath=savepath,
    )


if __name__ == "__main__":
    main()
