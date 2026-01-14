# training/train.py
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
from torch.optim import AdamW
from tqdm import tqdm

from data import get_mnist_dataloaders
from diffusion.transition import D3PMForward, make_noised_batch
from diffusion.losses import d3pm_loss  # expects the new K-generic losses.py
from models.unet_logits import UNetConfig, SmallUNetLogits


@torch.no_grad()
def macro_class_accuracy(x_true: torch.Tensor, x_pred: torch.Tensor, K: int) -> float:
    """
    Macro average over classes of per-class pixel accuracy:
        acc_k = P(pred==true | true==k)
    averaged over classes that appear in x_true.
    """
    x_true = x_true.view(-1)
    x_pred = x_pred.view(-1)
    accs = []
    for k in range(K):
        mask = (x_true == k)
        if mask.any():
            accs.append((x_pred[mask] == k).float().mean().item())
    return float(sum(accs) / len(accs)) if accs else 0.0


@torch.no_grad()
def batch_metrics(
    logits_x0: torch.Tensor,  # (B,K,H,W)
    x0: torch.Tensor,         # (B,1,H,W) long
    K: int,
) -> Dict[str, float]:
    pred = logits_x0.argmax(dim=1, keepdim=True)  # (B,1,H,W)
    pix_acc = (pred == x0).float().mean().item()
    macro_acc = macro_class_accuracy(x0, pred, K=K)
    return {"pixel_acc": pix_acc, "macro_acc": macro_acc}


@torch.no_grad()
def eval_fixed_timesteps(
    model: torch.nn.Module,
    forward: D3PMForward,
    x0: torch.Tensor,
    timesteps: List[int],
) -> Dict[str, float]:
    """
    Evaluate x0-prediction accuracy for a fixed x0 batch at a few timesteps.
    """
    out: Dict[str, float] = {}
    for tt in timesteps:
        xt = forward.sample_xt(x0, tt)
        t = torch.full((x0.shape[0],), tt, device=x0.device, dtype=torch.long)
        logits = model(xt, t)
        mets = batch_metrics(logits, x0, K=forward.K)
        for k, v in mets.items():
            out[f"{k}@t={tt}"] = v
    return out


def format_metrics(d: Dict[str, float], keys: List[str]) -> str:
    parts = []
    for k in keys:
        if k in d:
            parts.append(f"{k}: {d[k]:.4f}")
    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    # Discretization
    parser.add_argument("--K", type=int, default=4)

    # Diffusion
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.2)
    parser.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"])

    # Optim
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lambda_aux", type=float, default=1e-1)

    # Logging
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1)

    # Repro
    parser.add_argument("--seed", type=int, default=1998)

    # Save
    parser.add_argument("--out_ckpt", type=str, default="outputs/model.pt")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)

    # Data: now discretized into {0..K-1} inside data.py
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        K=args.K,
    )

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
    print("Using K, T:", forward.K, forward.T)

    # Model
    cfg = UNetConfig(
        in_channels=1,
        num_classes=args.K,   # K=4
        base_channels=64,
        time_emb_dim=64,
        time_hidden_dim=256,
        num_res_blocks=2,
    )
    model = SmallUNetLogits(cfg).to(device)
    optim = AdamW(model.parameters(), lr=args.lr)

    # Fixed eval batch (test)
    x0_eval, _ = next(iter(test_loader))
    x0_eval = x0_eval.to(device)  # (B,1,28,28) long in {0..K-1}

    probe_ts = [1, max(2, args.T // 10), args.T // 2, args.T]

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)

        running = {"loss": 0.0, "L_t-1": 0.0, "L_aux": 0.0}
        running_count = 0

        for x0, _ in pbar:
            x0 = x0.to(device)  # (B,1,H,W) long in {0..K-1}

            # Sample x_t and t
            xt, t = make_noised_batch(forward, x0)

            # Compute losses (new API computes logits internally)
            out = d3pm_loss(
                forward=forward,
                model=model,
                x0=x0,
                xt=xt,
                t=t,
                aux_weight=args.lambda_aux,
            )

            optim.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            global_step += 1
            running["loss"] += float(out.loss.item())
            running["L_t-1"] += float(out.L_tminus1.item())
            running["L_aux"] += float(out.L_aux.item())
            running_count += 1

            if global_step % args.log_every == 0:
                avg = {k: v / max(1, running_count) for k, v in running.items()}

                # Use metrics from the loss module (already K-generic)
                pix_acc = out.metrics.get("pix_acc", float("nan"))
                macro_acc = out.metrics.get("macro_acc", out.metrics.get("bal_acc", float("nan")))

                pbar.set_postfix_str(
                    f"loss {avg['loss']:.4f} | L_t-1 {avg['L_t-1']:.4f} | L_aux {avg['L_aux']:.4f} | "
                    f"pix_acc {pix_acc:.4f} macro_acc {macro_acc:.4f}"
                )
                running = {"loss": 0.0, "L_t-1": 0.0, "L_aux": 0.0}
                running_count = 0

        # Epoch eval probe
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                probe = eval_fixed_timesteps(model, forward, x0_eval, probe_ts)
            print("\n[Eval probe on fixed test batch]")
            keys = []
            for tt in probe_ts:
                keys += [f"pixel_acc@t={tt}", f"macro_acc@t={tt}"]
            print(format_metrics(probe, keys))
            print("")

    torch.save(model.state_dict(), args.out_ckpt)
    print(f"Saved checkpoint: {args.out_ckpt}")


if __name__ == "__main__":
    main()
