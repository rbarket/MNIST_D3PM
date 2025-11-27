# main.py
#
# Headless-safe MNIST sample viewer:
# - Never tries to open a GUI window
# - Saves preview image to disk instead

import torch
import matplotlib
matplotlib.use("Agg")       # <-- important: headless backend
import matplotlib.pyplot as plt


from data import get_mnist_dataloaders
from diffusion.transition import (
    linear_beta_schedule,
    build_all_transition_matrices,
    compute_cumulative_transition_matrices
)

def save_samples(images, labels, num=8, filename="samples.png"):
    """
    Saves a preview of several MNIST samples to a PNG file.
    Works in headless environments.
    """
    plt.figure(figsize=(10, 2))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(images[i].squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
        plt.title(f"{labels[i].item()}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved sample preview to {filename}")


def main():
    # 1) Load MNIST loaders
    train_loader, _ = get_mnist_dataloaders(
        batch_size=32,
        data_root="./data",
        threshold=0.5
    )

    x, y = next(iter(train_loader))

    print("=== MNIST Binary Data Sanity Check ===")
    print("Batch shape:", x.shape)
    print("Unique pixel values:", torch.unique(x))

    # Save sample images instead of showing them
    save_samples(x, y, filename="mnist_samples.png")

    T = 5 # timesteps
    betas = linear_beta_schedule(T)
    Qs = build_all_transition_matrices(betas)
    Qbar = compute_cumulative_transition_matrices(Qs)

    print("Q1:\n", Qs[0])
    
    print("Qbar[2]:\n", Qbar[1])
    print("Qbar[T]:\n", Qbar[T-1])


if __name__ == "__main__":
    main()
