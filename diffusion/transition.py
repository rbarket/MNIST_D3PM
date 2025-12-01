import torch
import torch.distributions as dist

# ---------- SCHEDULES ----------

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.5):
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.5)


# ---------- TRANSITION MATRICES ----------

def build_transition_matrix(beta_t):
    return torch.tensor([
        [1 - beta_t, beta_t],
        [beta_t,     1 - beta_t]
    ], dtype=torch.float32)


def build_all_transition_matrices(betas):
    return torch.stack([build_transition_matrix(b) for b in betas])  # [T,2,2]


def compute_cumulative_transition_matrices(Qs, device=None):
    T = Qs.shape[0]
    Qbar = torch.zeros_like(Qs)
    # Keep running product on the same device/dtype as Qs to avoid CPU/GPU mismatch
    cumulative = torch.eye(2, device=Qs.device, dtype=Qs.dtype)
    for t in range(T):
        cumulative = cumulative @ Qs[t]
        Qbar[t] = cumulative
    return Qbar  # [T,2,2]


# ---------- q(x_t | x_0) SAMPLING ----------

def _sample_q_xt_given_x0_scalar_t(x0, t, Qbar):
    """
    Internal helper for a single scalar t.
    x0: any shape, integer {0,1}
    """
    Qbar_t = Qbar[t]          # [2,2]
    probs = Qbar_t[x0]        # [...,2]
    xt = dist.Categorical(probs=probs).sample()
    return xt


def sample_q_xt_given_x0(x0, t, Qbar):
    """
    Sample x_t ~ q(x_t | x0) with either:
      - scalar t (int)
      - per-sample t: LongTensor of shape [B]

    x0: LongTensor [B,1,H,W] (for MNIST) or any shape
    t:  int  OR  LongTensor [B]
    Qbar: [T,2,2]
    """
    if isinstance(t, int):
        # old behavior: one t for whole batch
        return _sample_q_xt_given_x0_scalar_t(x0, t, Qbar)

    # t is tensor [B]: one timestep per sample
    assert t.dim() == 1, "t should be shape [B] when tensor"
    B = x0.shape[0]
    xt_list = []
    for i in range(B):
        xt_i = _sample_q_xt_given_x0_scalar_t(x0[i], int(t[i].item()), Qbar)
        xt_list.append(xt_i.unsqueeze(0))  # keep batch dim
    xt = torch.cat(xt_list, dim=0)
    return xt


# ---------- q(x_{t-1} | x_t, x_0) POSTERIOR ----------

def _compute_discrete_posterior_scalar_t(xt, x0, t, Qs, Qbar):
    """
    Internal helper for scalar t.
    xt, x0: same shape, integer {0,1}
    Returns: posterior [...,2]
    """
    if t <= 0:
        raise ValueError("t must be >= 1.")

    Q_t = Qs[t]          # [2,2]
    Qbar_tm1 = Qbar[t-1] # [2,2]

    # term1 = q(x_t | x_{t-1}=k) = Q_t[k, x_t]
    term1 = Q_t[:, xt]             # [2,...]
    term1 = term1.movedim(0, -1)   # [...,2]

    # term2 = q(x_{t-1}=k | x0) = Qbar_{t-1}[x0, k]
    term2 = Qbar_tm1[x0]           # [...,2]

    unnormalized = term1 * term2   # [...,2]
    posterior = unnormalized / (unnormalized.sum(dim=-1, keepdim=True) + 1e-20)
    return posterior               # [...,2]


def compute_discrete_posterior(xt, x0, t, Qs, Qbar):
    """
    Compute q(x_{t-1} | x_t, x0) with either:
      - scalar t (int)
      - per-sample t: LongTensor [B]

    xt, x0: [B,1,H,W]
    Returns: [B,1,H,W,2]
    """
    if isinstance(t, int):
        post = _compute_discrete_posterior_scalar_t(xt, x0, t, Qs, Qbar)
        return post  # [B,1,H,W,2]

    assert t.dim() == 1, "t should be shape [B] when tensor"
    B = x0.shape[0]
    post_list = []
    for i in range(B):
        post_i = _compute_discrete_posterior_scalar_t(
            xt[i], x0[i], int(t[i].item()), Qs, Qbar
        )  # [1,H,W,2] since xt[i] has shape [1,H,W]
        post_list.append(post_i.unsqueeze(0))  # add batch dim
    posterior = torch.cat(post_list, dim=0)   # [B,1,H,W,2]
    return posterior
