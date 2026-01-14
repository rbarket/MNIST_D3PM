from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F

from diffusion.transition import D3PMForward


def _gather_1d(buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Gather buf[t-1] for per-example integer t in [1..T].
    buf: (T,)
    t: (B,)
    returns: (B,)
    """
    if t.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"t must be int tensor, got {t.dtype}")
    return buf.gather(0, (t - 1).view(-1)).view(t.shape)


@torch.no_grad()
def q_posterior_xtm1_given_xt_x0(
    forward: D3PMForward,
    x0: torch.Tensor,                 # (B,1,H,W) long
    xt: torch.Tensor,                 # (B,1,H,W) long
    t: Union[int, torch.Tensor],      # int or (B,) long
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Exact forward posterior q(x_{t-1} | x_t, x_0) for the uniform D3PM kernel.

    Returns:
        probs: (B,1,H,W,K) float, sums to 1 over last dim.
    """
    if x0.dtype not in (torch.int32, torch.int64) or xt.dtype not in (torch.int32, torch.int64):
        raise TypeError("x0 and xt must be integer tensors.")
    if x0.shape != xt.shape:
        raise ValueError(f"x0 and xt must have same shape, got {x0.shape} vs {xt.shape}")
    if x0.min().item() < 0 or x0.max().item() >= forward.K:
        raise ValueError(f"x0 values must be in [0,{forward.K-1}]")
    if xt.min().item() < 0 or xt.max().item() >= forward.K:
        raise ValueError(f"xt values must be in [0,{forward.K-1}]")

    device = x0.device
    B = x0.shape[0]
    K = forward.K

    # Handle scalar t by expanding to (B,)
    if isinstance(t, int):
        if not (1 <= t <= forward.T):
            raise ValueError(f"t must be in [1,{forward.T}], got {t}")
        t_b = torch.full((B,), t, device=device, dtype=torch.long)
    else:
        t_b = t.to(device=device, dtype=torch.long)
        if t_b.shape != (B,):
            raise ValueError(f"t must have shape (B,), got {tuple(t_b.shape)}")
        if t_b.min().item() < 1 or t_b.max().item() > forward.T:
            raise ValueError(f"t must be in [1,{forward.T}], got [{t_b.min().item()},{t_b.max().item()}]")

    # If t == 1: x_{t-1} == x0 exactly (delta)
    # Return one-hot at x0.
    if torch.all(t_b == 1):
        return F.one_hot(x0, num_classes=K).to(torch.float32)  # (B,1,H,W,K)

    # For mixed t values, handle t==1 positions separately.
    # We'll compute generic posterior, then overwrite where t==1.
    # ---------------------------------------------------------

    # beta_t: (B,)
    beta_t = _gather_1d(forward.betas.to(device=device, dtype=torch.float32), t_b)  # (B,)
    beta = beta_t.view(B, *([1] * (x0.ndim - 1)))  # (B,1,1,1) broadcast to x0

    # fact2 = q(x_{t-1} | x0): (B,1,H,W,K) from forward closed form at time t-1
    # Note: forward.q_xt_given_x0_probs expects t in [1..T]. We want t-1 in [1..T-1].
    tm1 = (t_b - 1).clamp(min=1)
    fact2 = forward.q_xt_given_x0_probs(x0, tm1)  # (B,1,H,W,K)

    # fact1 = q(x_t | x_{t-1}=k) as a vector over k, given observed x_t
    # For uniform kernel:
    #   P(x_t=v | x_{t-1}=k) = beta/K + (1-beta) * 1[k==v]
    # Here v = x_t (per-pixel), and we construct a K-vector over k.
    base = (beta / float(K)).unsqueeze(-1).expand(*x0.shape, K).clone()  # (B,1,H,W,K)
    add = (1.0 - beta).unsqueeze(-1)  # (B,1,H,W,1)
    fact1 = base
    fact1.scatter_add_(dim=-1, index=xt.unsqueeze(-1), src=add.expand_as(xt.unsqueeze(-1)).to(fact1.dtype))

    # posterior ∝ fact1 * fact2
    unnorm = fact1 * fact2
    probs = unnorm / unnorm.sum(dim=-1, keepdim=True).clamp(min=eps)

    # overwrite t==1 elements with one-hot(x0)
    mask_t1 = (t_b == 1).view(B, *([1] * (x0.ndim - 1)), 1)  # (B,1,1,1,1)
    if mask_t1.any():
        probs_t1 = F.one_hot(x0, num_classes=K).to(probs.dtype)
        probs = torch.where(mask_t1, probs_t1, probs)

    return probs


@torch.no_grad()
def p_theta_xtm1_given_xt(
    forward: D3PMForward,
    logits_x0: torch.Tensor,          # (B,K,H,W) float
    xt: torch.Tensor,                 # (B,1,H,W) long
    t: torch.Tensor,                  # (B,) long
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Model reverse distribution p_theta(x_{t-1} | x_t).

    We compute:
        p_theta(x_{t-1}|x_t) ∝ q(x_t|x_{t-1}) * sum_{x0} q(x_{t-1}|x0) p_theta(x0|x_t)
    For the uniform kernel, the inner sum is available in closed form:
        q(x_{t-1}=k | x0=j) = base + alpha_bar_{t-1} * 1[k==j]
      => E_{p(x0|x_t)}[ q(x_{t-1}=k|x0) ] = base + alpha_bar_{t-1} * p(x0=k|x_t)

    Returns:
        probs: (B,1,H,W,K) float
    """
    if logits_x0.ndim != 4:
        raise ValueError(f"logits_x0 must be (B,K,H,W), got {tuple(logits_x0.shape)}")
    B, K, H, W = logits_x0.shape
    if K != forward.K:
        raise ValueError(f"logits_x0 K={K} must equal forward.K={forward.K}")
    if xt.shape != (B, 1, H, W):
        raise ValueError(f"xt must be (B,1,H,W) matching logits spatial dims, got {tuple(xt.shape)}")
    if t.shape != (B,):
        raise ValueError(f"t must be (B,), got {tuple(t.shape)}")

    device = logits_x0.device
    t = t.to(device=device, dtype=torch.long)
    if t.min().item() < 1 or t.max().item() > forward.T:
        raise ValueError(f"t must be in [1,{forward.T}], got [{t.min().item()},{t.max().item()}]")

    # p(x0 | x_t): (B,K,H,W)
    p_x0 = torch.softmax(logits_x0, dim=1)

    # beta_t and alpha_bar_{t-1}
    beta_t = _gather_1d(forward.betas.to(device=device, dtype=torch.float32), t)  # (B,)
    beta = beta_t.view(B, 1, 1, 1)  # broadcast over (1,H,W)

    # alpha_bar_{t-1}: for t==1, define alpha_bar_0 = 1.0 (no noise yet).
    # forward.alpha_bar expects t>=1, so handle t==1 separately.
    ab_tm1 = torch.empty((B,), device=device, dtype=torch.float32)
    mask_t1 = (t == 1)
    if mask_t1.any():
        ab_tm1[mask_t1] = 1.0
    if (~mask_t1).any():
        ab_tm1[~mask_t1] = forward.alpha_bar((t[~mask_t1] - 1).to(torch.long)).to(device=device, dtype=torch.float32)
    ab = ab_tm1.view(B, 1, 1, 1)  # broadcast over (1,H,W)

    # fact2 over k = x_{t-1}:
    # base = (1 - ab)/K, add ab * p_x0(k)
    base2 = (1.0 - ab) / float(K)                          # (B,1,1,1)
    fact2 = base2 + ab * p_x0.transpose(1, 0).transpose(0, 1)  # keep p_x0 shape (B,K,H,W)
    # fact2: (B,K,H,W)

    # fact1 over k given observed xt=v:
    # vector over k: beta/K + (1-beta) at k==v
    base1 = (beta / float(K)).expand(B, 1, H, W)  # (B,1,H,W)
    fact1 = base1.unsqueeze(-1).expand(B, 1, H, W, K).clone()
    add1 = (1.0 - beta).expand(B, 1, H, W).unsqueeze(-1)  # (B,1,H,W,1)
    fact1.scatter_add_(dim=-1, index=xt.unsqueeze(-1), src=add1.to(fact1.dtype))

    # Combine: unnorm = fact1 * fact2 (broadcast fact2 to (B,1,H,W,K))
    fact2_b = fact2.permute(0, 2, 3, 1).unsqueeze(1)  # (B,1,H,W,K)
    unnorm = fact1 * fact2_b
    probs = unnorm / unnorm.sum(dim=-1, keepdim=True).clamp(min=eps)

    # For t==1, x_{t-1}=x0, so p_theta(x0|x1) is the reverse:
    # set probs to p_x0 (as (B,1,H,W,K))
    if mask_t1.any():
        p_x0_b = p_x0.permute(0, 2, 3, 1).unsqueeze(1)  # (B,1,H,W,K)
        mask = mask_t1.view(B, 1, 1, 1, 1)
        probs = torch.where(mask, p_x0_b, probs)

    return probs
