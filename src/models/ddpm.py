"""DDPM (Ho et al., 2020) dengan U-Net kecil untuk citra 28x28x1.

Forward  q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I), jadwal beta linear 1e-4..0.02, T=1000.
Training : L_simple = || eps - eps_theta(x_t, t) ||^2
Sampling : x_{t-1} = 1/sqrt(a_t) * (x_t - b_t / sqrt(1 - abar_t) * eps_theta) + sigma_t z,  sigma_t^2 = b_t

U-Net (ukuran disesuaikan untuk GPU 4 GB):
  28x28 (64) -> 14x14 (128) -> 7x7 (128 + self-attention) -> naik kembali dengan skip connection.
Embedding waktu: sinusoidal -> MLP, disuntikkan ke setiap ResBlock.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ U-Net
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)   # (b, hw, hw)
        out = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        c1, c2 = ch, ch * 2
        tdim = ch * 4
        self.ch = ch
        self.time_mlp = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.conv_in = nn.Conv2d(1, c1, 3, padding=1)
        # encoder
        self.res0 = ResBlock(c1, c1, tdim)                        # 28
        self.down0 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)    # 28 -> 14
        self.res1 = ResBlock(c1, c2, tdim)                        # 14
        self.down1 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)    # 14 -> 7
        self.res2 = ResBlock(c2, c2, tdim)                        # 7
        self.attn2 = SelfAttention(c2)
        # bottleneck
        self.mid1 = ResBlock(c2, c2, tdim)
        self.mid_attn = SelfAttention(c2)
        self.mid2 = ResBlock(c2, c2, tdim)
        # decoder (concat skip)
        self.res_u2 = ResBlock(c2 + c2, c2, tdim)                 # 7
        self.attn_u2 = SelfAttention(c2)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c2, 3, padding=1))
        self.res_u1 = ResBlock(c2 + c2, c2, tdim)                 # 14
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c2, 3, padding=1))
        self.res_u0 = ResBlock(c2 + c1, c1, tdim)                 # 28
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.ch))
        h = self.conv_in(x)
        h0 = self.res0(h, temb)
        h1 = self.res1(self.down0(h0), temb)
        h2 = self.attn2(self.res2(self.down1(h1), temb))
        h = self.mid2(self.mid_attn(self.mid1(h2, temb)), temb)
        h = self.attn_u2(self.res_u2(torch.cat([h, h2], 1), temb))
        h = self.res_u1(torch.cat([self.up2(h), h1], 1), temb)
        h = self.res_u0(torch.cat([self.up1(h), h0], 1), temb)
        return self.out(h)


# ------------------------------------------------------------------ Diffusion
class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.model = model
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)
        self.register_buffer("sqrt_abar", abar.sqrt())
        self.register_buffer("sqrt_1m_abar", (1 - abar).sqrt())

    def q_sample(self, x0, t, noise):
        return self.sqrt_abar[t][:, None, None, None] * x0 + self.sqrt_1m_abar[t][:, None, None, None] * noise

    def loss(self, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(self.model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def p_sample(self, xt, t: int):
        tt = torch.full((xt.size(0),), t, device=xt.device, dtype=torch.long)
        eps = self.model(xt, tt)
        mean = (xt - self.betas[t] / self.sqrt_1m_abar[t] * eps) / self.alphas[t].sqrt()
        if t == 0:
            return mean
        return mean + self.betas[t].sqrt() * torch.randn_like(xt)

    @torch.no_grad()
    def sample(self, n: int, device, record_steps=None, generator=None):
        """Sampling penuh T langkah. record_steps: daftar t yang x_t-nya disimpan (trajektori)."""
        x = torch.randn(n, 1, 28, 28, generator=generator, device="cpu").to(device)
        traj = {}
        for t in reversed(range(self.T)):
            if record_steps is not None and t in record_steps:
                traj[t] = x.detach().cpu().clone()
            x = self.p_sample(x, t)
        x = x.clamp(-1, 1)
        if record_steps is not None:
            traj[-1] = x.detach().cpu().clone()     # -1 = x_0 akhir
            return x, traj
        return x


class EMA:
    """Exponential moving average bobot model (meningkatkan kualitas sampel DDPM)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)
