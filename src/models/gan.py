"""Vanilla GAN (MLP) dan DCGAN (konvolusional) untuk citra 28x28x1.

Output generator memakai Tanh -> rentang [-1, 1]. Diskriminator mengeluarkan logit
(tanpa Sigmoid) dan dilatih dengan BCEWithLogits (loss non-saturating).

DCGAN di slide dirancang untuk 64x64 RGB; di sini diadaptasi ke 28x28x1:
  G: z -> 128x7x7 -> ConvT -> 64x14x14 -> ConvT -> 1x28x28
  D: 1x28x28 -> Conv -> 64x14x14 -> Conv -> 128x7x7 -> Linear -> 1
Pedoman DCGAN dipertahankan: tanpa pooling (strided conv), BatchNorm di G dan D
(kecuali layer input D dan output G), ReLU di G, LeakyReLU(0.2) di D, init N(0, 0.02).
"""
import torch.nn as nn

from ..config import Z_DIM_GAN


def dcgan_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ------------------------------------------------------------------ Vanilla GAN
class MLPGenerator(nn.Module):
    def __init__(self, z_dim: int = Z_DIM_GAN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


# ------------------------------------------------------------------ DCGAN
class DCGenerator(nn.Module):
    def __init__(self, z_dim: int = Z_DIM_GAN, ngf: int = 64):
        super().__init__()
        self.ngf = ngf
        self.project = nn.Sequential(
            nn.Linear(z_dim, ngf * 2 * 7 * 7, bias=False),
            nn.BatchNorm1d(ngf * 2 * 7 * 7), nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),   # 7 -> 14
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1),                      # 14 -> 28
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.project(z).view(-1, self.ngf * 2, 7, 7)
        return self.net(h)


class DCDiscriminator(nn.Module):
    def __init__(self, ndf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),                      # 28 -> 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),                              # 14 -> 7
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(ndf * 2 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def build_gan(name: str, z_dim: int = Z_DIM_GAN):
    if name == "vanilla_gan":
        G, D = MLPGenerator(z_dim), MLPDiscriminator()
    elif name == "dcgan":
        G, D = DCGenerator(z_dim), DCDiscriminator()
    else:
        raise ValueError(name)
    G.apply(dcgan_init)
    D.apply(dcgan_init)
    return G, D
