import torch
import torch.nn.functional as F


def _p_theta_scalar_t_from_p_tilde(logits_x0, xt, t, Qs, Qbar):
    """
    Internal helper for scalar t.

    logits_x0: [B,K,H,W]
    xt:        [B,1,H,W]
    Returns:   [B,1,H,W,K] = p_theta(x_{t-1} | x_t)
    """
    B, K, H, W = logits_x0.shape

    # p̃_theta(x0 | x_t)
    p_tilde = F.softmax(logits_x0, dim=1)      # [B,K,H,W]

    Q_t = Qs[t]          # [K,K]
    Qbar_tm1 = Qbar[t-1] # [K,K]

    # flatten spatial dims
    p_tilde_flat = p_tilde.permute(0, 2, 3, 1).reshape(-1, K)  # [N,K]
    xt_flat = xt.view(-1).long()                               # [N]

    # term1[n,k] = Q_t[k, xt[n]]
    term1 = Q_t[:, xt_flat].permute(1, 0)                      # [N,K]

    # term2[n,:] = sum_{x0} p̃[n,x0] * Qbar_tm1[x0, :]
    term2_flat = p_tilde_flat @ Qbar_tm1                       # [N,K]

    unnormalized = term1 * term2_flat                          # [N,K]
    p_theta_flat = unnormalized / (unnormalized.sum(dim=1, keepdim=True) + 1e-20)

    p_theta = p_theta_flat.view(B, H, W, K).unsqueeze(1)       # [B,1,H,W,K]
    return p_theta


def compute_p_theta_xtminus1_given_xt_from_p_tilde(
    logits_x0, xt, t, Qs, Qbar
):
    """
    Vector-aware wrapper:
      - if t is int: old behavior
      - if t is LongTensor [B]: per-sample timestep

    logits_x0: [B,K,H,W]
    xt:        [B,1,H,W]
    Returns:   [B,1,H,W,K]
    """
    if isinstance(t, int):
        return _p_theta_scalar_t_from_p_tilde(logits_x0, xt, t, Qs, Qbar)

    assert t.dim() == 1, "t should be shape [B] when tensor"
    B = xt.shape[0]
    out_list = []
    for i in range(B):
        out_i = _p_theta_scalar_t_from_p_tilde(
            logits_x0[i:i+1], xt[i:i+1], int(t[i].item()), Qs, Qbar
        )  # [1,1,H,W,K]
        out_list.append(out_i)
    p_theta = torch.cat(out_list, dim=0)  # [B,1,H,W,K]
    return p_theta
