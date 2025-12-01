import torch
import matplotlib.pyplot as plt
from diffusion.transition import sample_q_xt_given_x0


@torch.no_grad()
def visualize_denoising(model, Qbar, x0_sample, t, device):
    """
    Visualize:
        - clean x0
        - noisy xt
        - predicted clean image x0_pred
    
    Args:
        model: your D3PM model
        Qbar: cumulative matrices [T, 2, 2]
        x0_sample: tensor [1,1,28,28]
        t: integer timestep
        device: cuda/cpu
    """
    model.eval()

    # Move sample to device
    x0 = x0_sample.to(device).long()         # [1,1,28,28]

    # Create noisy image x_t
    xt = sample_q_xt_given_x0(x0, torch.tensor([t], device=device), Qbar)

    # Predict logits for x0
    logits = model(xt, torch.tensor([t], device=device))  # [1,2,28,28]
    p_tilde = torch.softmax(logits, dim=1)
    x0_pred = p_tilde.argmax(dim=1, keepdim=True)         # [1,1,28,28]

    # ---- Plot ----
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    axs[0].imshow(x0[0,0].cpu(), cmap='gray')
    axs[0].set_title("Original x₀")

    axs[1].imshow(xt[0,0].cpu(), cmap='gray')
    axs[1].set_title(f"Noisy xₜ (t={t})")

    axs[2].imshow(x0_pred[0,0].cpu(), cmap='gray')
    axs[2].set_title("Denoised x₀̂")

    for ax in axs:
        ax.axis("off")

    plt.show()
