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
    # Load MNIST loaders
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


if __name__ == "__main__":
    main()
