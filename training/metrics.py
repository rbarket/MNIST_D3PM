import torch
import torch.nn.functional as F


# ---------------------------------------------------------
# Utility: Convert logits or probabilities to predicted labels
# ---------------------------------------------------------
def to_pred_labels(logits_or_probs):
    """
    Convert model output of shape [B, 2, H, W] or [B, 2, N]
    into integer predictions {0,1} of shape [B, H, W] or [B, N].
    """
    if logits_or_probs.dim() == 4:
        # [B,2,H,W] -> [B,H,W]
        return logits_or_probs.argmax(dim=1)
    elif logits_or_probs.dim() == 3:
        # [B,2,N] -> [B,N]
        return logits_or_probs.argmax(dim=1)
    else:
        raise ValueError(f"Unexpected shape: {logits_or_probs.shape}")


# ---------------------------------------------------------
# Pixel Accuracy (overall)
# ---------------------------------------------------------
def pixel_accuracy(x0, x0_pred):
    """
    Compute overall pixel accuracy across all pixels.

    Args:
        x0:      ground truth [B,1,H,W] or [B,H,W]
        x0_pred: predicted      same shape

    Returns:
        scalar accuracy (float)
    """
    x0 = x0.squeeze(1)
    correct = (x0_pred == x0).sum().item()
    total = x0.numel()
    return correct / total


# ---------------------------------------------------------
# Foreground Accuracy (x0 == 1)
# ---------------------------------------------------------
def foreground_accuracy(x0, x0_pred):
    """
    Accuracy restricted to foreground pixels (digit pixels, x0 == 1).
    Returns accuracy over all foreground pixels in the batch.
    """
    x0 = x0.squeeze(1)

    mask_fg = (x0 == 1)
    total_fg = mask_fg.sum().item()
    if total_fg == 0:
        return 0.0

    correct_fg = ((x0_pred == x0) & mask_fg).sum().item()
    return correct_fg / total_fg


# ---------------------------------------------------------
# Background Accuracy (x0 == 0)
# ---------------------------------------------------------
def background_accuracy(x0, x0_pred):
    """
    Accuracy restricted to background pixels (x0 == 0).
    """
    x0 = x0.squeeze(1)

    mask_bg = (x0 == 0)
    total_bg = mask_bg.sum().item()
    if total_bg == 0:
        return 0.0

    correct_bg = ((x0_pred == x0) & mask_bg).sum().item()
    return correct_bg / total_bg


# ---------------------------------------------------------
# Balanced Accuracy = (FG acc + BG acc) / 2
# ---------------------------------------------------------
def balanced_accuracy(x0, x0_pred):
    """
    Balanced accuracy between foreground and background.
    Helps avoid MNIST imbalance (86% background).
    """
    fg = foreground_accuracy(x0, x0_pred)
    bg = background_accuracy(x0, x0_pred)
    return 0.5 * (fg + bg)


# ---------------------------------------------------------
# Hamming Distance (normalized)
# ---------------------------------------------------------
def hamming_distance(x0, x0_pred):
    """
    Fraction of pixels that differ between x0 and x0_pred.
    (1 - accuracy), but sometimes easier to interpret.
    """
    x0 = x0.squeeze(1)
    diff = (x0_pred != x0).sum().item()
    total = x0.numel()
    return diff / total


# ---------------------------------------------------------
# Mean Absolute Error
# ---------------------------------------------------------
def mean_absolute_error(x0, x0_pred):
    """
    MAE between binary images.
    Useful because it's linear in pixel errors.
    """
    x0 = x0.squeeze(1).float()
    x0_pred = x0_pred.float()
    return torch.abs(x0 - x0_pred).mean().item()


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------
def confusion_matrix(x0, x0_pred):
    """
    Returns a 2x2 confusion matrix:
        [[TN, FP],
         [FN, TP]]
    for binary data.
    """
    x0 = x0.squeeze(1)

    TN = ((x0 == 0) & (x0_pred == 0)).sum().item()
    FP = ((x0 == 0) & (x0_pred == 1)).sum().item()
    FN = ((x0 == 1) & (x0_pred == 0)).sum().item()
    TP = ((x0 == 1) & (x0_pred == 1)).sum().item()

    return torch.tensor([[TN, FP],
                         [FN, TP]])


# ---------------------------------------------------------
# Main Denoising Evaluation Wrapper
# ---------------------------------------------------------
def evaluate_denoising(model, x0, t, Qbar, device):
    """
    High-level evaluation for a chosen timestep t.
    Computes all metrics at once.

    Args:
        model:  your denoiser model
        x0:     ground-truth clean sample [B,1,H,W]
        t:      timestep(s)
        Qbar:   cumulative transitions
        device: device

    Returns dict of all metrics.
    """
    from diffusion.transition import sample_q_xt_given_x0  # avoid circular import

    model.eval()
    with torch.no_grad():
        x0 = x0.to(device).long()

        # sample noisy xt
        xt = sample_q_xt_given_x0(x0, t, Qbar).to(device)

        # model prediction logits
        logits = model(xt, t)
        x0_pred = to_pred_labels(logits)

        # metrics
        return {
            "pixel_acc": pixel_accuracy(x0, x0_pred),
            "fg_acc": foreground_accuracy(x0, x0_pred),
            "bg_acc": background_accuracy(x0, x0_pred),
            "balanced_acc": balanced_accuracy(x0, x0_pred),
            "hamming": hamming_distance(x0, x0_pred),
            "mae": mean_absolute_error(x0, x0_pred),
            "confusion": confusion_matrix(x0, x0_pred),
        }
