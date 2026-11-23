"""Konfigurasi global: seed, dataset, hyperparameter bersama.

Semua model memakai dataset, split, seed, dan batch size yang SAMA agar
perbedaan hasil mencerminkan inductive bias tiap model (aturan standardisasi tugas).
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

SEED = 42
DATASET = "FashionMNIST"      # dataset tunggal untuk semua model
BATCH_SIZE = 128
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1   # sisanya (0.1) = test
IMG_SIZE = 28
LATENT_DIM = 64               # dim laten AE / Sparse AE / VAE (sama dgn MLP z_MLP)
Z_DIM_GAN = 100               # prior noise GAN / DCGAN


@dataclass
class AEConfig:
    epochs: int = 30
    lr: float = 1e-3
    sparse_hidden: int = 256   # lapisan sparse (Sparse AE)
    sparse_lambda: float = 1e-3  # bobot penalti L1 pada aktivasi laten
    vae_beta: float = 1.0      # bobot KL pada VAE
