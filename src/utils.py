"""Utilitas: seed, device, checkpoint, pencatatan waktu/latensi."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import SEED


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # reproduktibilitas (sedikit lebih lambat)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


@torch.no_grad()
def measure_latency(fn, n_runs: int = 50, warmup: int = 10) -> float:
    """Rata-rata latensi (ms) untuk memanggil fn() satu kali.

    fn: callable tanpa argumen yang menjalankan satu batch (mis. 64 sampel).
    Memakai cuda.synchronize agar pengukuran GPU valid.
    """
    use_cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000.0
