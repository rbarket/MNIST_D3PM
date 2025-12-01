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
    Train the D3PM model for one epoch using Equation (5):

        L = KL(q || p_theta) + lambda_aux * (-log p̃_theta(x0 | x_t))

    Uses:
        - per-sample timesteps t
        - x0 parameterization
        - true discrete posterior
        - reverse kernel p_theta(x_{t-1}|x_t)

    Returns:
        stats: dict of averaged metrics from the epoch
    """

    model.train()

    total_loss = 0.0
    total_kl = 0.0
    total_aux = 0.0
    step_count = 0

    for x0, _ in dataloader:
        # ---------- 1. Move data ----------
        x0 = x0.to(device).long()      # [B,1,H,W]
        B = x0.shape[0]

        # ---------- 2. Sample random timestep per sample ----------
        t = torch.randint(1, T, (B,), device=device)   # [B]

        # ---------- 3. Sample x_t ~ q(x_t | x0) ----------
        xt = sample_q_xt_given_x0(x0, t, Qbar)         # [B,1,H,W]

        # ---------- 4. Model forward: logits for x0 ----------
        logits_x0 = model(xt, t)                       # [B,2,H,W]

        # ---------- 5. True posterior q(x_{t-1} | x_t, x0) ----------
        q_posterior = compute_discrete_posterior(
            xt, x0, t, Qs, Qbar
        )                                              # [B,1,H,W,2]

        # ---------- 6. Reverse kernel p_theta(x_{t-1} | x_t) ----------
        p_theta = compute_p_theta_xtminus1_given_xt_from_p_tilde(
            logits_x0, xt, t, Qs, Qbar
        )                                              # [B,1,H,W,2]

        # ---------- 7. KL(q || p_theta) ----------
        log_q = torch.log(q_posterior + 1e-20)
        log_p = torch.log(p_theta + 1e-20)
        kl = (q_posterior * (log_q - log_p)).sum(dim=-1).mean()

        # ---------- 8. Auxiliary CE term (-log p̃_theta(x0|x_t)) ----------
        p_tilde = F.softmax(logits_x0, dim=1)          # [B,2,H,W]
        p_x0 = p_tilde.gather(1, x0)                   # [B,1,H,W]
        aux_ce = -(torch.log(p_x0 + 1e-20)).mean()

        # ---------- 9. Final Eq. (5) loss ----------
        loss = kl + lambda_aux * aux_ce

        # ---------- 10. Backprop ----------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---------- 11. Logging ----------
        total_loss += loss.item()
        total_kl += kl.item()
        total_aux += aux_ce.item()
        step_count += 1

    # ---------- 12. Epoch summary ----------
    return {
        "loss": total_loss / step_count,
        "kl": total_kl / step_count,
        "aux_ce": total_aux / step_count,
        "steps": step_count,
    }
