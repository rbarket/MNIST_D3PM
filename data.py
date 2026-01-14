# data.py
import torch
from torch.utils.data import DataLoader
from typing import Optional
from torchvision import datasets, transforms


class Discretize(object):
    """
    Discretize a grayscale tensor in [0,1] into K integer levels {0,1,...,K-1}.

    Mapping:
        x_int = round(x * (K-1))  -> clamp to [0, K-1]
    """
    def __init__(self, K: int = 4):
        if K < 2:
            raise ValueError(f"K must be >= 2, got {K}.")
        self.K = K

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img is float32 tensor in [0,1], shape (1,H,W)
        x = (img * (self.K - 1)).round().clamp(0, self.K - 1)
        return x.long()  # -> integers {0,...,K-1}


def get_mnist_dataloaders(
    batch_size: int = 128,
    data_root: str = "./data",
    num_workers: int = 4,
    K: int = 4,
    train_subset_size: Optional[int] = None,
    test_subset_size: Optional[int] = None,
    subset_seed: int = 0,
):
    """
    Returns MNIST train/test loaders with discretized images in {0,...,K-1}.

    Parameters:
        batch_size (int): Batch size for DataLoader.
        data_root (str): MNIST directory.
        num_workers (int): DataLoader workers.
        K (int): Number of discrete pixel values (default 4).
        train_subset_size (Optional[int]): If set, randomly sample this many training examples.
        test_subset_size (Optional[int]): If set, randomly sample this many test examples.
        subset_seed (int): RNG seed used for subset sampling.

    Returns:
        train_loader, test_loader
    """

    transform = transforms.Compose([
        transforms.ToTensor(),     # -> [0,1] float32, shape (1,H,W)
        Discretize(K=K),           # -> {0,...,K-1} long
    ])

    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    # Optionally downsample datasets for quick experiments
    if train_subset_size is not None:
        gen = torch.Generator().manual_seed(subset_seed)
        indices = torch.randperm(len(train_dataset), generator=gen)[:train_subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    if test_subset_size is not None:
        gen = torch.Generator().manual_seed(subset_seed)
        indices = torch.randperm(len(test_dataset), generator=gen)[:test_subset_size]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
