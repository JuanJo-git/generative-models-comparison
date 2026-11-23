"""Data loader: Fashion-MNIST dengan split 80/10/10 dan seed tetap.

Piksel disimpan di [0, 1] (ToTensor). Model berbasis Tanh (GAN/DCGAN/DDPM)
mengonversi ke [-1, 1] lewat to_minus1_1 / to_zero1 agar metrik dihitung
pada skala yang sama.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import (BATCH_SIZE, DATA_DIR, SEED, TRAIN_FRAC, VAL_FRAC)


def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def to_zero1(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0


def get_datasets():
    tfm = transforms.ToTensor()  # [0, 1], shape (1, 28, 28)
    # Gabungkan train+test resmi (70.000) lalu split ulang 80/10/10.
    train_full = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test_full = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tfm)
    full = torch.utils.data.ConcatDataset([train_full, test_full])

    n = len(full)
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(SEED)
    return random_split(full, [n_train, n_val, n_test], generator=gen)


def get_loaders(batch_size: int = BATCH_SIZE, num_workers: int = 0):
    train_ds, val_ds, test_ds = get_datasets()
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=False)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train, val, test
