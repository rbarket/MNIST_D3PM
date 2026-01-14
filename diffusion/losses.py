# diffusion/losses.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from diffusion.transition import D3PMForward
from diffusion.posterior import q_posterior_xtm1_given_xt_x0, p_theta_xtm1_given_xt


@torch.no_grad()
def pixel_accuracy(x_true: torch.Tensor, x_pred: torch.Tensor) -> float:
    return (x_true == x_pred).float().mean().item()


@torch.no_grad()
def macro_class_accuracy(x_true: torch.Tensor, x_pred: torch.Tensor, K: int) -> float:
    """
    Macro average over classes of per-class pixel accuracy:
        acc_k = P(pred==true | true==k)
        macro = mean_k acc_k over classes that appear in x_true.
    """
    x_true = x_true.view(-1)
    x_pred = x_pred.view(-1)
    accs = []
    for k in range(K):
        mask = (x_true == k)
        if mask.any():
            accs.append((x_pred[mask] == k).float().mean().item())
    if len(accs) == 0:
        return 0.0
    return float(sum(accs) / len(accs))


def l_aux_ce(
    logits_x0: torch.Tensor,  # (B,K,H,W)
    x0: torch.Tensor,         # (B,1,H,W) long
) -> torch.Tensor:
    """
    Auxiliary CE term: -log p_theta(x0 | xt) (per pixel).
    """
    if logits_x0.ndim != 4:
        raise ValueError(f"logits_x0 must be (B,K,H,W), got {tuple(logits_x0.shape)}")
    if x0.ndim != 4 or x0.shape[1] != 1:
        raise ValueError(f"x0 must be (B,1,H,W), got {tuple(x0.shape)}")

    target = x0.squeeze(1).long()  # (B,H,W)
    return F.cross_entropy(logits_x0, target, reduction="mean")


def l_tminus1_vb(
    forward: D3PMForward,
    logits_x0: torch.Tensor,  # (B,K,H,W)
    x0: torch.Tensor,         # (B,1,H,W) long
    xt: torch.Tensor,         # (B,1,H,W) long
    t: torch.Tensor,          # (B,) long
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    KL(q(x_{t-1}|x_t,x0) || p_theta(x_{t-1}|x_t)) averaged over pixels and batch.
    """
    q = q_posterior_xtm1_given_xt_x0(forward, x0, xt, t, eps=eps)          # (B,1,H,W,K)
    p = p_theta_xtm1_given_xt(forward, logits_x0, xt, t, eps=eps)          # (B,1,H,W,K)

    # KL(q||p) = sum q * (log q - log p)
    kl = q * (torch.log(q.clamp(min=eps)) - torch.log(p.clamp(min=eps)))
    return kl.sum(dim=-1).mean()  # sum over K, mean over pixels/batch


@dataclass
class LossOutput:
    loss: torch.Tensor
    L_tminus1: torch.Tensor
    L_aux: torch.Tensor
    metrics: Dict[str, float]


def d3pm_loss(
    forward: D3PMForward,
    model: torch.nn.Module,
    x0: torch.Tensor,             # (B,1,H,W) long in [0..K-1]
    xt: torch.Tensor,             # (B,1,H,W) long
    t: torch.Tensor,              # (B,) long
    *,
    aux_weight: float = 0.001,
) -> LossOutput:
    """
    Combined objective:
        L = L_{t-1} + aux_weight * L_aux

    Where:
        L_{t-1} is the VB/KL term at timestep t (with special-casing t==1 inside posterior)
        L_aux is the CE on x0 prediction
    """
    logits_x0 = model(xt, t)  # expected (B,K,H,W)

    Lvb = l_tminus1_vb(forward, logits_x0, x0, xt, t)
    Laux = l_aux_ce(logits_x0, x0)
    L = Lvb + aux_weight * Laux

    # Metrics: use argmax as x0 prediction
    x0_hat = logits_x0.argmax(dim=1, keepdim=True).long()  # (B,1,H,W)
    pix_acc = pixel_accuracy(x0, x0_hat)
    macro_acc = macro_class_accuracy(x0, x0_hat, K=forward.K)

    metrics = {
        "loss": float(L.detach().item()),
        "L_t-1": float(Lvb.detach().item()),
        "L_aux": float(Laux.detach().item()),
        "pix_acc": pix_acc,
        "macro_acc": macro_acc,
        "bal_acc": macro_acc,  # keep name for backward-compat with your logs
    }

    return LossOutput(loss=L, L_tminus1=Lvb, L_aux=Laux, metrics=metrics)
