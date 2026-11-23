"""Data loader: Fashion-MNIST dengan split 80/10/10 dan seed tetap.

Seluruh dataset (70.000 citra) dimuat sekali ke memori sebagai tensor float [0, 1]
agar training jauh lebih cepat daripada transformasi PIL per sampel.
Pembagian split identik dengan versi sebelumnya (random_split dengan seed yang sama).

Model berbasis Tanh (GAN/DCGAN/DDPM) mengonversi ke [-1, 1] lewat to_minus1_1 /
to_zero1 agar metrik tetap dihitung pada skala yang sama.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets

from .config import BATCH_SIZE, DATA_DIR, SEED, TRAIN_FRAC, VAL_FRAC

CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def to_zero1(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0


def get_datasets():
    # Gabungkan train+test resmi (70.000) lalu split ulang 80/10/10.
    tr = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    te = datasets.FashionMNIST(DATA_DIR, train=False, download=True)
    x = torch.cat([tr.data, te.data]).float().div_(255.0).unsqueeze(1)  # (70000,1,28,28)
    y = torch.cat([tr.targets, te.targets])
    full = TensorDataset(x, y)

    n = len(full)
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(SEED)
    return random_split(full, [n_train, n_val, n_test], generator=gen)


def get_loaders(batch_size: int = BATCH_SIZE, num_workers: int = 0):
    train_ds, val_ds, test_ds = get_datasets()
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train, val, test
