import torch
import torch.nn.functional as F

from diffusion.transition import (
    sample_q_xt_given_x0,
    compute_discrete_posterior,
)
from diffusion.reverse import (
    compute_p_theta_xtminus1_given_xt_from_p_tilde,
)


def train_one_epoch(
    model,
    optimizer,
    dataloader,
    Qs,
    Qbar,
    T,
    device,
    lambda_aux=0.001,
):
    """
    Train the D3PM model for one epoch with a single-timestep Monte Carlo
    estimator of

        L_vb ≈ L_{t-1} + L_0

    plus an auxiliary denoising term:

        L = (L_{t-1} + L_0) + lambda_aux * L_aux

    where
        - L_{t-1} = KL(q(x_{t-1}|x_t,x0) || p_theta(x_{t-1}|x_t))   for t > 1
        - L_0     = -log p_theta(x0 | x1)                          for t = 1
        - L_aux   = -log p̃_theta(x0 | x_t)  (auxiliary CE at all t)

    We do this in a *single* model/transition pass per batch for speed,
    and separate L_{t-1} vs L_0 by masking on t.
    """

    model.train()

    total_loss = 0.0
    total_L_tminus1 = 0.0
    total_L0 = 0.0
    total_aux = 0.0
    step_count = 0

    eps = 1e-12

    for x0, _ in dataloader:
        # ---------- 1. Move data ----------
        x0 = x0.to(device).long()      # [B,1,H,W]
        B = x0.shape[0]

        # ---------- 2. Sample random timestep per sample ----------
        # t in {1, ..., T-1}; ELBO sum over t>=2 is approximated via t>1,
        # and t=1 is used for the L0 term.
        t = torch.randint(1, T, (B,), device=device)   # [B]

        # ---------- 3. Sample x_t ~ q(x_t | x0) ----------
        xt = sample_q_xt_given_x0(x0, t, Qbar)         # [B,1,H,W]

        # ---------- 4. Model forward: logits for x0 ----------
        logits_x0 = model(xt, t)                       # [B,K,H,W]

        # ---------- 5. True posterior q(x_{t-1} | x_t, x0) ----------
        q_posterior = compute_discrete_posterior(
            xt, x0, t, Qs, Qbar
        )                                              # [B,1,H,W,K]

        # ---------- 6. Reverse kernel p_theta(x_{t-1} | x_t) ----------
        p_theta = compute_p_theta_xtminus1_given_xt_from_p_tilde(
            logits_x0, xt, t, Qs, Qbar
        )                                              # [B,1,H,W,K]

        # ---------- 7. Per-pixel KL for L_{t-1} ----------
        log_q = torch.log(q_posterior + eps)           # [B,1,H,W,K]
        log_p = torch.log(p_theta + eps)               # [B,1,H,W,K]
        kl_per = (q_posterior * (log_q - log_p)).sum(dim=-1)  # [B,1,H,W]

        # ---------- 8. Per-pixel NLL for L_0 and aux ----------
        # p̃_theta(x0 | x_t) from logits_x0
        log_p_tilde = F.log_softmax(logits_x0, dim=1)      # [B,K,H,W]
        log_p_x0 = log_p_tilde.gather(1, x0)               # [B,1,H,W]
        nll_per = -log_p_x0                                # [B,1,H,W]

        # ---------- 9. Mask by timestep: t>1 -> L_{t-1}, t==1 -> L_0 ----------
        mask_L0 = (t == 1).view(B, 1, 1, 1).float()        # [B,1,1,1]
        mask_Lt = 1.0 - mask_L0

        # L_{t-1}: only contributions from t>1
        L_tminus1 = (mask_Lt * kl_per).mean()

        # L_0: only contributions from t==1, using NLL
        L0 = (mask_L0 * nll_per).mean()

        # Variational bound term (single-timestep MC estimator)
        vb_loss = L_tminus1 + L0

        # ---------- 10. Auxiliary CE: -E[log p̃_theta(x0|x_t)] over all t ----------
        aux_ce = nll_per.mean()

        # ---------- 11. Final loss ----------
        loss = vb_loss + lambda_aux * aux_ce

        # ---------- 12. Backprop ----------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---------- 13. Logging ----------
        total_loss += loss.item()
        total_L_tminus1 += L_tminus1.item()
        total_L0 += L0.item()
        total_aux += aux_ce.item()
        step_count += 1

    # ---------- 14. Epoch summary ----------
    return {
        "loss": total_loss / step_count,
        "L_tminus1": total_L_tminus1 / step_count,
        "L0": total_L0 / step_count,
        "aux_ce": total_aux / step_count,
        "steps": step_count,
    }
