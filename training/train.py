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


def eval_fixed_timesteps(
    model: torch.nn.Module,
    forward: D3PMForward,
    x0: torch.Tensor,
    timesteps: List[int],
    y: torch.Tensor,  # REQUIRED
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for tt in timesteps:
        xt = forward.sample_xt(x0, tt)
        t = torch.full((x0.shape[0],), tt, device=x0.device, dtype=torch.long)
        logits = model(xt, t, y)  # REQUIRED
        mets = batch_metrics(logits, x0, K=forward.K)
        for k, v in mets.items():
            out[f"{k}@t={tt}"] = v
    return out


@torch.no_grad()
def eval_loss_over_loader(
    model: torch.nn.Module,
    forward: D3PMForward,
    data_loader: torch.utils.data.DataLoader,
    *,
    aux_weight: float,
    device: torch.device,
) -> Dict[str, float]:
    totals = {"loss": 0.0, "L_t-1": 0.0, "L_aux": 0.0}
    count = 0
    for x0, y in data_loader:
        x0 = x0.to(device)
        y = y.to(device)

        xt, t = make_noised_batch(forward, x0)
        out = d3pm_loss(
            forward=forward,
            model=model,
            x0=x0,
            xt=xt,
            t=t,
            aux_weight=aux_weight,
            y=y,  # REQUIRED
        )
        totals["loss"] += float(out.loss.item())
        totals["L_t-1"] += float(out.L_tminus1.item())
        totals["L_aux"] += float(out.L_aux.item())
        count += 1
    return {k: v / max(1, count) for k, v in totals.items()}


def format_metrics(d: Dict[str, float], keys: List[str]) -> str:
    parts = []
    for k in keys:
        if k in d:
            parts.append(f"{k}: {d[k]:.4f}")
    return "  ".join(parts)


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    aim_run_hash: str | None,
) -> None:
    ckpt = {
        "state_dict": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
    }
    if aim_run_hash:
        ckpt["aim_run_hash"] = aim_run_hash
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[int, int, bool, str | None]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
        if optim is not None and "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        aim_run_hash = ckpt.get("aim_run_hash")
        if aim_run_hash is not None:
            aim_run_hash = str(aim_run_hash)
        return epoch, global_step, True, aim_run_hash

    model.load_state_dict(ckpt, strict=True)
    return 0, 0, False, None


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
    parser.add_argument("--aim_run_name", type=str, default=None, help="Aim run name")

    # Repro
    parser.add_argument("--seed", type=int, default=1998)

    # Save
    parser.add_argument("--out_ckpt", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume (weights-only or full checkpoint).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs when --out_ckpt is set (0 disables).",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.out_ckpt:
        out_dir = os.path.dirname(args.out_ckpt)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

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

    start_epoch = 1
    global_step = 0
    resume_aim_hash = None
    if args.resume:
        loaded_epoch, loaded_step, has_optim, resume_aim_hash = load_checkpoint(
            args.resume,
            model=model,
            optim=optim,
            device=device,
        )
        if loaded_epoch > 0:
            start_epoch = loaded_epoch + 1
        global_step = loaded_step
        if has_optim:
            print(
                f"Resumed from checkpoint: {args.resume} "
                f"(epoch {loaded_epoch}, step {loaded_step})"
            )
        else:
            print(f"Loaded weights from: {args.resume}")

    try:
        from aim import Run
    except Exception as exc:
        raise RuntimeError(
            "Aim is required for training. Install aim to track metrics."
        ) from exc

    if resume_aim_hash:
        try:
            run = Run(repo=".", run_hash=resume_aim_hash)
        except TypeError:
            try:
                run = Run(repo=".", hash=resume_aim_hash)
            except TypeError:
                print(
                    "Warning: Aim Run does not support resuming by hash; "
                    "starting a new run."
                )
                run = Run(repo=".", experiment="mnist-d3pm")
        except Exception:
            print("Warning: Failed to resume Aim run; starting a new run.")
            run = Run(repo=".", experiment="mnist-d3pm")
    else:
        run = Run(repo=".", experiment="mnist-d3pm")

    if args.aim_run_name:
        run.name = args.aim_run_name
    run["hparams"] = vars(args)
    if args.resume:
        run["resume_from"] = args.resume

    if start_epoch > args.epochs:
        print(
            f"Checkpoint epoch {start_epoch - 1} >= requested epochs {args.epochs}. "
            "Nothing to do."
        )
        run.close()
        return

    x0_eval, y_eval = next(iter(test_loader))
    x0_eval = x0_eval.to(device)
    y_eval = y_eval.to(device)

    probe_ts = [1, max(2, args.T // 10), args.T // 2, args.T]

    last_saved_epoch = 0
    last_epoch = start_epoch - 1
    aim_run_hash = getattr(run, "hash", None)
    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)

        running = {"loss": 0.0, "L_t-1": 0.0, "L_aux": 0.0}
        running_count = 0

        for x0, y in pbar:
            x0 = x0.to(device)
            y = y.to(device)

            xt, t = make_noised_batch(forward, x0)

            out = d3pm_loss(
                forward=forward,
                model=model,
                x0=x0,
                xt=xt,
                t=t,
                aux_weight=args.lambda_aux,
                y=y,  # REQUIRED
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
                run.track(
                    avg["loss"],
                    name="loss",
                    step=global_step,
                    epoch=epoch,
                    context={"subset": "train"},
                )
                run.track(
                    avg["L_t-1"],
                    name="L_t-1",
                    step=global_step,
                    epoch=epoch,
                    context={"subset": "train"},
                )
                run.track(
                    avg["L_aux"],
                    name="L_aux",
                    step=global_step,
                    epoch=epoch,
                    context={"subset": "train"},
                )
                running = {"loss": 0.0, "L_t-1": 0.0, "L_aux": 0.0}
                running_count = 0

        # Epoch eval probe
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                probe = eval_fixed_timesteps(model, forward, x0_eval, probe_ts, y=y_eval)
                test_loss = eval_loss_over_loader(
                    model,
                    forward,
                    test_loader,
                    aux_weight=args.lambda_aux,
                    device=device,
                )
            print("\n[Eval probe on fixed test batch]")
            keys = []
            for tt in probe_ts:
                keys += [f"pixel_acc@t={tt}", f"macro_acc@t={tt}"]
            print(format_metrics(probe, keys))
            print(
                f"test loss {test_loss['loss']:.4f} | "
                f"L_t-1 {test_loss['L_t-1']:.4f} | L_aux {test_loss['L_aux']:.4f}"
            )
            run.track(
                test_loss["loss"],
                name="loss",
                step=global_step,
                epoch=epoch,
                context={"subset": "test"},
            )
            run.track(
                test_loss["L_t-1"],
                name="L_t-1",
                step=global_step,
                epoch=epoch,
                context={"subset": "test"},
            )
            run.track(
                test_loss["L_aux"],
                name="L_aux",
                step=global_step,
                epoch=epoch,
                context={"subset": "test"},
            )
            print("")

        if args.out_ckpt and args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                args.out_ckpt,
                model=model,
                optim=optim,
                epoch=epoch,
                global_step=global_step,
                args=args,
                aim_run_hash=aim_run_hash,
            )
            last_saved_epoch = epoch
            print(f"Saved checkpoint: {args.out_ckpt}")

    if args.out_ckpt and last_epoch >= start_epoch and last_saved_epoch != last_epoch:
        save_checkpoint(
            args.out_ckpt,
            model=model,
            optim=optim,
            epoch=last_epoch,
            global_step=global_step,
            args=args,
            aim_run_hash=aim_run_hash,
        )
        print(f"Saved checkpoint: {args.out_ckpt}")
    run.close()


if __name__ == "__main__":
    main()
