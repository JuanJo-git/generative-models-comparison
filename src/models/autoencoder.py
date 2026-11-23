"""Vanilla AE, Sparse AE, dan VAE (Sesi 06).

Semua memakai input/output di [0, 1] (Sigmoid) dan encoder MLP:
784 -> 256 -> 64 (sama dengan ekstraktor fitur MLP Classifier agar perbandingan adil).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LATENT_DIM


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, in_dim: int = 784, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, out_dim: int = 784, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class VanillaAE(nn.Module):
    """Autoencoder deterministik; loss = MSE piksel."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    def loss(self, x):
        x_hat = self(x)
        l = F.mse_loss(x_hat, x)
        return l, {"recon": l.item()}


class SparseAE(nn.Module):
    """Sparse AE: lapisan sparse ber-ReLU (256) + penalti L1 pada aktivasinya.

    x -> 784 -> 256(sparse, ReLU) -> 64 (bottleneck) -> 256 -> 784
    Loss = MSE + lambda * ||h||_1, dengan h = aktivasi lapisan sparse (256).
    """

    def __init__(self, latent_dim: int = LATENT_DIM, sparse_hidden: int = 256, sparse_lambda: float = 1e-3):
        super().__init__()
        self.sparse_lambda = sparse_lambda
        self.enc_sparse = nn.Sequential(nn.Flatten(), nn.Linear(784, sparse_hidden), nn.ReLU())
        self.enc_bottleneck = nn.Linear(sparse_hidden, latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode_full(self, x):
        h = self.enc_sparse(x)
        z = self.enc_bottleneck(h)
        return z, h

    def encode(self, x):
        return self.encode_full(x)[0]

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    def loss(self, x):
        z, h = self.encode_full(x)
        x_hat = self.decode(z)
        recon = F.mse_loss(x_hat, x)
        l1 = h.abs().mean()   # rata-rata |h| (ternormalisasi agar lambda tidak bergantung ukuran)
        total = recon + self.sparse_lambda * l1
        return total, {"recon": recon.item(), "l1": l1.item()}


class VAE(nn.Module):
    """VAE: encoder -> (mu, logvar), reparameterization, decoder.

    Loss = BCE rekonstruksi (jumlah per piksel) + beta * KL(q(z|x) || N(0, I)),
    dirata-ratakan per sampel. Dilaporkan juga recon dalam MSE untuk perbandingan.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.latent_dim = latent_dim
        self.enc_body = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode_dist(self, x):
        h = self.enc_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def encode(self, x):
        """Untuk rekonstruksi/interpolasi deterministik, pakai mean."""
        return self.encode_dist(x)[0]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode_dist(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def reconstruct(self, x):
        return self.decode(self.encode(x))

    @torch.no_grad()
    def sample(self, n: int, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

    def loss(self, x):
        x_hat, mu, logvar = self(x)
        bce = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total = bce + self.beta * kl
        mse = F.mse_loss(x_hat, x).item()
        return total, {"recon": bce.item(), "kl": kl.item(), "mse": mse}
