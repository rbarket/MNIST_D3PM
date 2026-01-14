# training/test_posterior.py
from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusion.transition import D3PMForward, sample_timesteps_uniform
from diffusion.posterior import (
    q_posterior_xtm1_given_xt_x0,
    p_theta_xtm1_given_xt,
)


def assert_close(a: torch.Tensor, b: torch.Tensor, atol=1e-5, rtol=1e-5, msg=""):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = (a - b).abs().max().item()
        raise AssertionError(f"{msg} (max_abs={max_abs})")


@torch.no_grad()
def test_q_posterior_normalization_and_shape(device: torch.device):
    print("[TEST] q_posterior: shape + normalization")

    forward = D3PMForward.from_linear_schedule(K=2, T=200, beta_start=1e-4, beta_end=0.2, device=device)
    B, H, W = 32, 28, 28

    # Random binary x0
    x0 = torch.randint(0, 2, (B, 1, H, W), device=device, dtype=torch.long)

    # Random t in [1..T]
    t = sample_timesteps_uniform(B, forward.T, device=device)

    # Sample xt from the forward marginal
    xt = forward.sample_xt(x0, t)

    qpost = q_posterior_xtm1_given_xt_x0(forward, x0, xt, t)  # (B,1,H,W,K)

    assert qpost.shape == (B, 1, H, W, 2), f"Unexpected shape: {qpost.shape}"
    # Normalization over K
    sums = qpost.sum(dim=-1)
    assert_close(sums, torch.ones_like(sums), atol=1e-6, msg="qpost does not sum to 1 over K")
    # Non-negativity
    assert (qpost >= -1e-8).all().item(), "qpost has negative probabilities"
    print("  ✓ passed")


@torch.no_grad()
def test_q_posterior_t1_is_delta_at_x0(device: torch.device):
    print("[TEST] q_posterior: t=1 gives delta at x0 (x_{t-1}=x0)")

    forward = D3PMForward.from_linear_schedule(K=2, T=200, beta_start=1e-4, beta_end=0.2, device=device)
    B, H, W = 16, 28, 28

    x0 = torch.randint(0, 2, (B, 1, H, W), device=device, dtype=torch.long)
    t = 1

    # xt sampled from q(x1|x0)
    xt = forward.sample_xt(x0, t)

    qpost = q_posterior_xtm1_given_xt_x0(forward, x0, xt, t)  # distribution over x0

    # Expected: one-hot at x0
    # Convert x0 to one-hot over K and compare to qpost
    x0_oh = F.one_hot(x0.squeeze(1), num_classes=2).unsqueeze(1).to(torch.float32)  # (B,1,H,W,2)

    assert_close(qpost, x0_oh, atol=1e-6, msg="At t=1, posterior is not delta at x0")
    print("  ✓ passed")


@torch.no_grad()
def test_p_theta_normalization_and_shape(device: torch.device):
    print("[TEST] p_theta: shape + normalization")

    forward = D3PMForward.from_linear_schedule(K=2, T=200, beta_start=1e-4, beta_end=0.2, device=device)
    B, H, W = 8, 28, 28

    x0 = torch.randint(0, 2, (B, 1, H, W), device=device, dtype=torch.long)
    t = sample_timesteps_uniform(B, forward.T, device=device)
    xt = forward.sample_xt(x0, t)

    # Random logits (as if from a model)
    logits = torch.randn(B, 2, H, W, device=device)

    pth = p_theta_xtm1_given_xt(forward, logits, xt, t)

    assert pth.shape == (B, 1, H, W, 2), f"Unexpected shape: {pth.shape}"
    sums = pth.sum(dim=-1)
    assert_close(sums, torch.ones_like(sums), atol=1e-6, msg="p_theta does not sum to 1 over K")
    assert (pth >= -1e-8).all().item(), "p_theta has negative probabilities"
    print("  ✓ passed")


@torch.no_grad()
def test_p_theta_extreme_logits_matches_qpost_for_that_x0(device: torch.device):
    print("[TEST] p_theta: extreme logits -> matches qpost(x0_hat)")

    forward = D3PMForward.from_linear_schedule(K=2, T=200, beta_start=1e-4, beta_end=0.2, device=device)
    B, H, W = 4, 28, 28

    # xt can be arbitrary; pick something random
    xt = torch.randint(0, 2, (B, 1, H, W), device=device, dtype=torch.long)
    t = sample_timesteps_uniform(B, forward.T, device=device)

    # Force p_theta(x0=0|xt) ~ 1 everywhere
    logits = torch.empty(B, 2, H, W, device=device)
    logits[:, 0, :, :] = 50.0
    logits[:, 1, :, :] = -50.0

    pth = p_theta_xtm1_given_xt(forward, logits, xt, t)

    # Should match q(x_{t-1}|xt, x0_hat=0)
    x0_hat0 = torch.zeros(B, 1, H, W, device=device, dtype=torch.long)
    qpost0 = q_posterior_xtm1_given_xt_x0(forward, x0_hat0, xt, t)

    assert_close(pth, qpost0, atol=1e-5, msg="p_theta does not match qpost for extreme logits")
    print("  ✓ passed")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_q_posterior_normalization_and_shape(device)
    test_q_posterior_t1_is_delta_at_x0(device)
    test_p_theta_normalization_and_shape(device)
    test_p_theta_extreme_logits_matches_qpost_for_that_x0(device)

    print("\nAll posterior tests passed ✅")


if __name__ == "__main__":
    main()
