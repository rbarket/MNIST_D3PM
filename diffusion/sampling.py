# diffusion/sampling.py
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusion.transition import D3PMForward
from diffusion.posterior import p_theta_xtm1_given_xt


def _ensure_b1hw_shape(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """
    Accept (B,H,W) or (B,1,H,W). Return (B,1,H,W).
    """
    if len(shape) == 3:
        b, h, w = shape
        return (b, 1, h, w)
    if len(shape) == 4 and shape[1] == 1:
        return shape  # already (B,1,H,W)
    raise ValueError(f"Expected shape (B,H,W) or (B,1,H,W), got {shape}")


@torch.no_grad()
def sample_prior(
    forward: D3PMForward,
    shape: Tuple[int, ...],
    *,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample x_T from the D3PM prior. For the uniform transition matrix used here,
    the stationary distribution is uniform over {0,...,K-1} per pixel, so we use
    Uniform categorical.

    Args:
        forward: D3PMForward (contains K)
        shape: (B,1,H,W) or (B,H,W)
    Returns:
        xT: long tensor (B,1,H,W) with values in {0,...,K-1}
    """
    b1hw = _ensure_b1hw_shape(shape)
    device = device if device is not None else torch.device("cpu")

    if forward.K == 2:
        # Bernoulli(0.5) in {0,1}
        u = torch.rand(b1hw, device=device, dtype=torch.float32, generator=generator)
        return (u < 0.5).to(torch.long)

    return torch.randint(
        low=0,
        high=forward.K,
        size=b1hw,
        device=device,
        dtype=torch.long,
        generator=generator,
    )


@torch.no_grad()
def sample_categorical(
    probs: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample integer categorical values from probs.

    Args:
        probs: (..., K) probabilities (must sum to 1 on last dim).
    Returns:
        samples: (...) long tensor in {0,...,K-1}
    """
    if probs.ndim < 1:
        raise ValueError("probs must have at least 1 dimension")

    K = probs.shape[-1]
    if K == 2:
        p1 = probs[..., 1]
        u = torch.rand(p1.shape, device=p1.device, dtype=p1.dtype, generator=generator)
        return (u < p1).to(torch.long)

    flat = probs.reshape(-1, K).clamp(min=1e-12)
    flat = flat / flat.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(flat, num_samples=1, replacement=True, generator=generator).squeeze(-1)
    return idx.view(*probs.shape[:-1]).to(torch.long)


@torch.no_grad()
def reverse_step(
    model: nn.Module,
    forward: D3PMForward,
    xt: torch.Tensor,
    t: Union[int, torch.Tensor],
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    One reverse step: sample x_{t-1} ~ p_theta(x_{t-1} | x_t).

    Args:
        model: outputs logits for x0: (B,K,H,W) given (xt, t_tensor)
        forward: D3PMForward
        xt: (B,1,H,W) long in {0..K-1}
        t: int scalar or (B,) long tensor, timestep in [1..T]
    Returns:
        x_prev: (B,1,H,W) long
    """
    if xt.ndim != 4 or xt.shape[1] != 1:
        raise ValueError(f"xt must be (B,1,H,W), got {tuple(xt.shape)}")
    device = xt.device
    B = xt.shape[0]

    # Ensure t is (B,) tensor for the model
    if isinstance(t, int):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
    else:
        if t.ndim != 1 or t.shape[0] != B:
            raise ValueError(f"t must be shape (B,), got {tuple(t.shape)} for B={B}")
        t_tensor = t.to(device=device, dtype=torch.long)

    # Model predicts logits for x0
    logits_x0 = model(xt, t_tensor)  # (B,K,H,W)

    # Convert to p_theta(x_{t-1} | x_t) via D3PM mixture
    p_xtm1 = p_theta_xtm1_given_xt(forward, logits_x0, xt, t_tensor)  # (B,1,H,W,K)

    # Sample x_{t-1}
    x_prev = sample_categorical(p_xtm1, generator=generator)  # (B,1,H,W)
    return x_prev


@torch.no_grad()
def sample_loop(
    model: nn.Module,
    forward: D3PMForward,
    *,
    batch_size: int = 16,
    image_size: Tuple[int, int] = (28, 28),
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
    return_trajectory: bool = False,
    trajectory_steps: Optional[List[int]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]]:
    """
    Full sampling loop (Step 5):
        x_T ~ p(x_T)
        for t = T..1:
            x_{t-1} ~ p_theta(x_{t-1} | x_t)

    Args:
        return_trajectory: if True, also return selected intermediate x_t states.
        trajectory_steps: list of timesteps to store (e.g. [T, T//2, 1]).
                          If None and return_trajectory=True, stores [T, T//2, 1].
    Returns:
        x0: (B,1,H,W) long
        (optional) trajectory: list of (t, x_t_copy) with x_t tensors on CPU
    """
    device = device if device is not None else next(model.parameters()).device
    H, W = image_size

    # Start from x_T
    xt = sample_prior(
        forward,
        (batch_size, 1, H, W),
        device=device,
        generator=generator,
    )

    traj: List[Tuple[int, torch.Tensor]] = []
    if return_trajectory:
        if trajectory_steps is None:
            trajectory_steps = [forward.T, max(1, forward.T // 2), 1]
        trajectory_steps = sorted(set(trajectory_steps), reverse=True)

        if forward.T in trajectory_steps:
            traj.append((forward.T, xt.detach().cpu()))

    # Reverse chain
    for t in range(forward.T, 0, -1):
        xt = reverse_step(model, forward, xt, t, generator=generator)  # now xt is x_{t-1}

        if return_trajectory and trajectory_steps is not None and t - 1 in trajectory_steps:
            traj.append((t - 1, xt.detach().cpu()))

    x0 = xt  # after finishing t=1 step, xt is x_0
    if return_trajectory:
        return x0, traj
    return x0
