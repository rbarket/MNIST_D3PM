import torch
import torch.nn.functional as F

from diffusion.transition import sample_q_xt_given_x0, compute_discrete_posterior
from diffusion.reverse import compute_p_theta_xtminus1_given_xt_from_p_tilde


def d3pm_loss_eq5(
    model,
    x0,
    Qs,
    Qbar,
    T,
    lambda_aux=0.001,
):
    """
    Implementation of Eq. (5) with per-sample t:

    L_tau = L_vb + tau * E[-log p̃_theta(x0 | x_t)]

    x0: [B,1,H,W], integers {0,1}
    """
    device = x0.device
    B = x0.shape[0]

    # 1. sample one timestep per sample: t[i] in [1,T-1]
    t = torch.randint(1, T, (B,), device=device)  # [B]

    # 2. sample x_t for each sample
    xt = sample_q_xt_given_x0(x0, t, Qbar)       # [B,1,H,W]

    # 3. model predicts logits for x0 given x_t and per-sample t
    logits_x0 = model(xt, t)                     # [B,K,H,W]

    # 4. true posterior q(x_{t-1}|x_t,x0) for each sample
    q_posterior = compute_discrete_posterior(xt, x0, t, Qs, Qbar)  # [B,1,H,W,2]

    # 5. model reverse kernel p_theta(x_{t-1}|x_t) via x0-param
    p_theta = compute_p_theta_xtminus1_given_xt_from_p_tilde(
        logits_x0, xt, t, Qs, Qbar
    )  # [B,1,H,W,2]

    # 6. KL(q || p_theta)
    log_q = torch.log(q_posterior + 1e-20)
    log_p = torch.log(p_theta + 1e-20)
    kl = (q_posterior * (log_q - log_p)).sum(dim=-1).mean()  # scalar

    # 7. Aux term: -log p̃_theta(x0 | x_t)
    p_tilde = F.softmax(logits_x0, dim=1)                  # [B,2,H,W]
    x0_labels = x0.long()                                  # [B,1,H,W]
    p_x0 = p_tilde.gather(1, x0_labels)                    # [B,1,H,W]
    aux_ce = -(torch.log(p_x0 + 1e-20)).mean()             # scalar

    loss = kl + lambda_aux * aux_ce

    diagnostics = {
        "kl": kl.detach(),
        "aux_ce": aux_ce.detach(),
        "loss": loss.detach(),
        "t_mean": t.float().mean().detach(),
    }
    return loss, diagnostics
