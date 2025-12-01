import torch
import torch.nn.functional as F
from diffusion.transition import sample_q_xt_given_x0

@torch.no_grad()
def denoising_accuracy(model, Qbar, dataloader, T, device, t_eval=None, num_batches=10):
    """
    Estimate pixel-wise accuracy of predicting x0 from xt.
    """
    model.eval()
    if t_eval is None:
        t_eval = T // 2

    total_correct = 0
    total_pixels = 0

    for i, (x0, _) in enumerate(dataloader):
        if i >= num_batches:
            break

        x0 = x0.to(device).long()          # [B,1,H,W]
        B, _, H, W = x0.shape

        # single t for now or per-sample, both fine for eval
        t = torch.full((B,), t_eval, device=device, dtype=torch.long)
        xt = sample_q_xt_given_x0(x0, t, Qbar)

        logits = model(xt, t)              # [B,2,H,W]
        p_tilde = F.softmax(logits, dim=1)
        x0_pred = p_tilde.argmax(dim=1, keepdim=True)   # [B,1,H,W]

        correct = (x0_pred == x0).sum().item()
        total_correct += correct
        total_pixels += B * H * W

    return total_correct / total_pixels
