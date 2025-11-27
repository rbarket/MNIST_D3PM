import torch
import torch.distributions as dist

# -----------------------------------------------
# Step 1: Define β_t (noise schedule)
# -----------------------------------------------
def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.5):
    """
    Linear schedule for beta_t values.
    
    Args:
        T: number of diffusion steps
        beta_start: initial noise value
        beta_end: final noise value (max=0.5 for binary categories)
    
    Returns:
        betas: Tensor of shape [T]
    """
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T, s=0.008):
    """
    Cosine schedule adapted from continuous diffusion (Nichol & Dhariwal),
    but works fine for discrete too.
    """
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0.0001, 0.5)   # ensure within valid range
    return betas


# -----------------------------------------------
# Step 2: Build Q_t for K=2 (binary categories)
# -----------------------------------------------
def build_transition_matrix(beta_t):
    """
    Construct a 2x2 transition matrix Q_t for binary diffusion.
    
    Q_t = [[1 - beta_t, beta_t],
           [beta_t, 1 - beta_t]]

    Args:
        beta_t: scalar float or tensor
    
    Returns:
        Q_t: Tensor of shape [2, 2]
    """
    Q_t = torch.tensor([
        [1 - beta_t, beta_t],
        [beta_t, 1 - beta_t]
    ], dtype=torch.float32)

    return Q_t


def build_all_transition_matrices(betas):
    """
    Build a stack of all Q_t for t in 1..T
    
    Args:
        betas: Tensor of shape [T]
    
    Returns:
        Q: Tensor of shape [T, 2, 2]
    """
    Qs = torch.stack([build_transition_matrix(beta) for beta in betas])
    return Qs


def compute_cumulative_transition_matrices(Qs):
    """
    Compute cumulative transition matrices:
        Q_bar[t] = Q1 @ Q2 @ ... @ Qt

    Args:
        Qs: Tensor of shape [T, 2, 2] representing Q1..QT

    Returns:
        Q_bar: Tensor of shape [T, 2, 2]
    """
    T = Qs.shape[0]
    Q_bar = torch.zeros_like(Qs)

    # Start cumulative product with Q1
    cumulative = torch.eye(2)  # identity matrix for initial product
    for t in range(T):
        cumulative = cumulative @ Qs[t]      # multiply in sequence
        Q_bar[t] = cumulative

    return Q_bar


def sample_q_xt_given_x0(x0, t, Qbar):
    """
    Sample x_t ~ q(x_t | x0) using the cumulative transition matrix Qbar[t].

    Args:
        x0: Tensor of shape [B, 1, H, W] with values 0 or 1
        t: integer timestep (0 <= t < T)
        Qbar: Tensor of shape [T, 2, 2]

    Returns:
        x_t: Tensor of shape [B, 1, H, W], values in {0,1}
    """
    # Qbar_t: shape [2, 2]
    Qbar_t = Qbar[t]     # transition for timestep t

    # Flatten for vectorized distribution sampling
    x0_flat = x0.view(-1)          # shape [B*H*W]
    BHW = x0_flat.shape[0]

    # Gather the probability vector for each pixel
    # Example: if x0_flat[i] = 0 → Qbar_t[0], else Qbar_t[1]
    probs = Qbar_t[x0_flat]        # shape [B*H*W, 2]

    # Create categorical distribution
    categorical = dist.Categorical(probs=probs)

    # Sample x_t for each pixel
    xt_flat = categorical.sample()

    # Reshape back to image format
    xt = xt_flat.view_as(x0)

    return xt