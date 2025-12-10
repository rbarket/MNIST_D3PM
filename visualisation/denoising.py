import torch
import matplotlib.pyplot as plt
from diffusion.transition import sample_q_xt_given_x0


@torch.no_grad()
def visualize_denoising(model, Qbar, x0_sample, t, device):
    """
    Visualize denoising at a chosen timestep and at the final timestep:
        - clean x0
        - noisy x_t and predicted x0 at that t
        - noisy x_T and predicted x0 at T
    
    Args:
        model: your D3PM model
        Qbar: cumulative matrices [T, 2, 2]
        x0_sample: tensor [1,1,28,28]
        t: integer timestep
        device: cuda/cpu
    """
    model.eval()
    T_max = Qbar.shape[0] - 1  # last valid timestep index

    # Move sample to device
    x0 = x0_sample.to(device).long()         # [1,1,28,28]

    # ---- Middle timestep ----
    xt_mid = sample_q_xt_given_x0(x0, t, Qbar)
    logits_mid = model(xt_mid, torch.tensor([t], device=device))  # [1,2,28,28]
    p_tilde_mid = torch.softmax(logits_mid, dim=1)
    x0_pred_mid = p_tilde_mid.argmax(dim=1, keepdim=True)         # [1,1,28,28]

    # ---- Final timestep (T) ----
    xt_final = sample_q_xt_given_x0(x0, T_max, Qbar)
    logits_final = model(xt_final, torch.tensor([T_max], device=device))
    p_tilde_final = torch.softmax(logits_final, dim=1)
    x0_pred_final = p_tilde_final.argmax(dim=1, keepdim=True)

    # ---- Plot ----
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))

    # Row 0: chosen timestep
    axs[0,0].imshow(x0[0,0].cpu(), cmap='gray')
    axs[0,0].set_title("Original x₀")

    axs[0,1].imshow(xt_mid[0,0].cpu(), cmap='gray')
    axs[0,1].set_title(f"Noisy xₜ (t={t})")

    axs[0,2].imshow(x0_pred_mid[0,0].cpu(), cmap='gray')
    axs[0,2].set_title("Denoised x₀̂")

    # Row 1: final timestep
    axs[1,0].imshow(x0[0,0].cpu(), cmap='gray')
    axs[1,0].set_title("Original x₀")

    axs[1,1].imshow(xt_final[0,0].cpu(), cmap='gray')
    axs[1,1].set_title(f"Noisy x_T (t={T_max})")

    axs[1,2].imshow(x0_pred_final[0,0].cpu(), cmap='gray')
    axs[1,2].set_title("Denoised x₀̂ from x_T")

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    plt.show()
