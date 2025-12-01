import torch
from torch.utils.data import DataLoader
from typing import Optional
from torchvision import datasets, transforms


class Binarize(object):
    """
    Convert grayscale tensor to binary (0 or 1).
    Threshold default is 0.5.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, img):
        # img is float32 tensor in [0,1]
        return (img > self.threshold).long()   # -> integers {0,1}


def get_mnist_dataloaders(
    batch_size: int = 128,
    data_root: str = "./data",
    num_workers: int = 4,
    threshold: float = 0.5,
    train_subset_size: Optional[int] = None,
    test_subset_size: Optional[int] = None,
    subset_seed: int = 0,
):
    """
    Returns MNIST train/test loaders with binary images.

    Parameters:
        batch_size (int): Batch size for DataLoader.
        data_root (str): MNIST directory.
        num_workers (int): DataLoader workers.
        threshold (float): Binarization threshold.
        train_subset_size (Optional[int]): If set, randomly sample this many training examples.
        test_subset_size (Optional[int]): If set, randomly sample this many test examples.
        subset_seed (int): RNG seed used for subset sampling.

    Returns:
        train_loader, test_loader
    """

    transform = transforms.Compose([
        transforms.ToTensor(),        # -> [0,1] float32
        Binarize(threshold),          # -> {0,1}
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
