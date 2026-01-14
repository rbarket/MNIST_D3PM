# diffusion/transition.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch


def linear_beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.2,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Linear schedule for beta_t, t=1..T.

    Returns:
        betas: (T,) tensor with values in (0, 1).
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    betas = torch.linspace(beta_start, beta_end, T, device=device, dtype=dtype)
    # Guardrails: valid stochastic matrix requires 0 <= beta_t <= 1
    betas = betas.clamp(min=0.0, max=1.0)
    return betas

def cosine_beta_schedule(
    T: int,
    s: float = 0.008,
    max_beta: float = 0.999,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Cosine schedule via an alpha_bar(t) curve, then convert to betas.

    Uses:
        alpha_bar(t) = cos^2( (t/T + s) / (1+s) * pi/2 )

    Then:
        beta_t = 1 - alpha_bar(t) / alpha_bar(t-1)  for t=1..T

    Returns:
        betas: (T,) tensor with values in (0, 1).
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")

    # steps: 0..T
    steps = torch.arange(T + 1, device=device, dtype=dtype)
    t = steps / float(T)

    # alpha_bar in [0..T]
    alpha_bar = torch.cos(((t + s) / (1.0 + s)) * torch.pi / 2.0) ** 2
    # normalize so alpha_bar[0] == 1
    alpha_bar = alpha_bar / alpha_bar[0].clamp(min=1e-12)

    # betas: length T, for t=1..T
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1].clamp(min=1e-12))

    # clamp for numerical stability / valid stochastic transitions
    betas = betas.clamp(min=0.0, max=max_beta)
    return betas


@dataclass
class D3PMForward:
    """
    Forward (noising) process for D3PM with the *uniform* transition matrix:

        Q_t = (1 - beta_t) I + beta_t * (1/K) * 11^T

    For this particular Q_t family, the marginal q(x_t | x_0) has a closed form:

        \bar{Q}_t = prod_s (1 - beta_s) * I + (1 - prod_s (1 - beta_s)) * (1/K) * 11^T

    so for a pixel value v in {0..K-1}:
        P(x_t = v | x_0=v)   = alpha_bar_t + (1 - alpha_bar_t)/K
        P(x_t != v | x_0=v)  = (1 - alpha_bar_t)/K

    This avoids explicitly forming KxK matrices during sampling.

    Timesteps are 1-indexed: t in {1, ..., T}.
    """
    K: int
    betas: torch.Tensor  # (T,)
    alpha_bars: torch.Tensor  # (T,)

    @classmethod
    def from_linear_schedule(
        cls,
        *,
        K: int = 2,
        T: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "D3PMForward":
        betas = linear_beta_schedule(
            T=T, beta_start=beta_start, beta_end=beta_end, device=device, dtype=dtype
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)  # (T,)
        return cls(K=K, betas=betas, alpha_bars=alpha_bars)

    @classmethod
    def from_cosine_schedule(
        cls,
        *,
        K: int = 2,
        T: int = 500,
        s: float = 0.008,
        max_beta: float = 0.999,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "D3PMForward":
        betas = cosine_beta_schedule(
            T=T, s=s, max_beta=max_beta, device=device, dtype=dtype
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)  # (T,)
        return cls(K=K, betas=betas, alpha_bars=alpha_bars)


    @property
    def T(self) -> int:
        return int(self.betas.shape[0])

    def to(self, device: Union[str, torch.device]) -> "D3PMForward":
        device = torch.device(device)
        self.betas = self.betas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def alpha_bar(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Returns alpha_bar_t = prod_{s=1..t} (1 - beta_s).
        """
        if isinstance(t, int):
            if not (1 <= t <= self.T):
                raise ValueError(f"t must be in [1, {self.T}], got {t}.")
            return self.alpha_bars[t - 1]
        if not torch.is_tensor(t):
            raise TypeError(f"t must be int or torch.Tensor, got {type(t)}.")
        if t.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"t must be integer tensor, got dtype={t.dtype}.")
        if t.min().item() < 1 or t.max().item() > self.T:
            raise ValueError(f"t must be in [1, {self.T}], got range [{t.min().item()}, {t.max().item()}].")
        return self.alpha_bars.gather(0, (t - 1).view(-1)).view(t.shape)

    @torch.no_grad()
    def q_xt_given_x0_probs(
        self,
        x0: torch.Tensor,
        t: Union[int, torch.Tensor],
        *,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Compute probabilities for q(x_t | x_0) per pixel.

        Args:
            x0: Long tensor of shape (B, 1, H, W) or (B, H, W), values in {0,...,K-1}.
            t: int scalar or int tensor of shape (B,) (broadcasted across pixels).
        Returns:
            probs: Float tensor of shape (*x0.shape, K) giving categorical probabilities.
                   (If x0 is (B,1,H,W), probs is (B,1,H,W,K)).
        """
        if x0.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"x0 must be integer tensor, got dtype={x0.dtype}.")
        if x0.min().item() < 0 or x0.max().item() >= self.K:
            raise ValueError(f"x0 values must be in [0, {self.K-1}].")

        device = x0.device
        alpha_bar_t = self.alpha_bar(t).to(device=device, dtype=torch.float32)

        # Shape alpha_bar_t for broadcasting: (B, 1, 1, 1, 1) or scalar -> broadcast.
        if torch.is_tensor(alpha_bar_t):
            if alpha_bar_t.ndim == 0:
                ab = alpha_bar_t
            else:
                # Expect (B,) typically; reshape to match batch dim of x0
                if alpha_bar_t.shape[0] != x0.shape[0]:
                    raise ValueError(
                        f"If t is a tensor, expected shape (B,), got {tuple(alpha_bar_t.shape)} "
                        f"for batch size B={x0.shape[0]}."
                    )
                # broadcast over remaining dims
                ab = alpha_bar_t.view(-1, *([1] * (x0.ndim - 1)))
        else:
            ab = torch.tensor(alpha_bar_t, device=device, dtype=torch.float32)

        # Closed form:
        # For each pixel, probs = (1 - ab)/K for all classes, then add ab to the x0 class.
        base = (1.0 - ab) / float(self.K)  # broadcastable to x0
        # Build probs with last dimension K
        probs = base.unsqueeze(-1).expand(*x0.shape, self.K).clone()
        # Ensure src has the same shape as index (i.e., x0.unsqueeze(-1))
        if torch.is_tensor(ab):
            if ab.ndim == 0:
                ab_full = torch.full_like(x0, float(ab.item()), dtype=probs.dtype)
            else:
                # ab is (B, 1, 1, 1) broadcastable to x0
                ab_full = ab.expand_as(x0).to(dtype=probs.dtype)
        else:
            ab_full = torch.full_like(x0, float(ab), dtype=probs.dtype)

        probs.scatter_add_(
            dim=-1,
            index=x0.unsqueeze(-1),
            src=ab_full.unsqueeze(-1),
        )
        # Numerical normalization guard
        probs = probs.clamp(min=0.0)
        probs = probs / (probs.sum(dim=-1, keepdim=True).clamp(min=eps))
        return probs

    @torch.no_grad()
    def sample_xt(
        self,
        x0: torch.Tensor,
        t: Union[int, torch.Tensor],
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_0) using the uniform D3PM forward process.

        Args:
            x0: Long tensor (B,1,H,W) or (B,H,W) with values in {0,...,K-1}.
            t:  int scalar in [1,T] or int tensor of shape (B,) in [1,T].
        Returns:
            xt: Long tensor with same shape as x0.
        """
        probs = self.q_xt_given_x0_probs(x0, t)  # (*x0, K)

        if self.K == 2:
            # For K=2, categorical reduces to Bernoulli on class "1"
            p1 = probs[..., 1]
            u = torch.rand(
                p1.shape,
                device=p1.device,
                dtype=p1.dtype,
                generator=generator,
            )

            xt = (u < p1).to(dtype=torch.long)
            return xt

        # Generic categorical sampling (more expensive for large K)
        flat_probs = probs.reshape(-1, self.K)
        xt_flat = torch.multinomial(flat_probs, num_samples=1, replacement=True, generator=generator).squeeze(-1)
        xt = xt_flat.view(*x0.shape).to(dtype=torch.long)
        return xt

    @torch.no_grad()
    def sample_step(
        self,
        xtm1: torch.Tensor,
        t: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample one forward step: x_t ~ q(x_t | x_{t-1}) using Q_t.

        This is optional (you can rely entirely on sample_xt(x0,t)), but useful for debugging.

        For uniform Q_t:
            P(stay same) = (1 - beta_t) + beta_t/K
            P(change to any other class) = beta_t/K

        Args:
            xtm1: Long tensor of shape (B,1,H,W) or (B,H,W)
            t: timestep in [1,T]
        """
        if not (1 <= t <= self.T):
            raise ValueError(f"t must be in [1, {self.T}], got {t}.")
        beta = self.betas[t - 1].to(device=xtm1.device, dtype=torch.float32)

        if self.K == 2:
            # With K=2, "change to other class" is just flip.
            p_flip = beta / 2.0
            u = torch.rand_like(xtm1.to(torch.float32), generator=generator)
            flip = (u < p_flip).to(torch.long)
            return (xtm1 ^ flip)  # xor flips 0<->1

        # Generic K: sample new class from uniform w.p. beta, else keep.
        u = torch.rand_like(xtm1.to(torch.float32), generator=generator)
        do_uniform = (u < beta).to(torch.bool)
        uni = torch.randint(low=0, high=self.K, size=xtm1.shape, device=xtm1.device, generator=generator)
        xt = torch.where(do_uniform, uni, xtm1)
        return xt


def sample_timesteps_uniform(
    batch_size: int,
    T: int,
    device: Optional[torch.device] = None,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample t ~ Uniform({1,...,T}) for each element in the batch.
    Returns int64 tensor of shape (B,).
    """
    return torch.randint(low=1, high=T + 1, size=(batch_size,), device=device, dtype=torch.int64, generator=generator)


def make_noised_batch(
    forward: D3PMForward,
    x0: torch.Tensor,
    *,
    t: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience helper for training:
        - samples timesteps t (if not provided)
        - samples x_t ~ q(x_t | x_0)

    Args:
        forward: D3PMForward instance
        x0: long tensor (B,1,H,W) or (B,H,W)
        t: optional (B,) int64 tensor in [1,T]
    Returns:
        xt, t
    """
    if t is None:
        t = sample_timesteps_uniform(x0.shape[0], forward.T, device=x0.device, generator=generator)
    xt = forward.sample_xt(x0, t, generator=generator)
    return xt, t
