"""Metrik rekonstruksi: MSE, PSNR, SSIM (semua pada skala [0, 1]).

FID dihitung terpisah (src/fid.py) dengan protokol yang sama untuk semua model.
"""
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


def mse_per_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y, reduction="none").flatten(1).mean(1)


def psnr_per_image(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = mse_per_image(x, y).clamp_min(1e-10)
    return 10.0 * torch.log10(max_val ** 2 / mse)


def ssim_batch(x: torch.Tensor, y: torch.Tensor) -> float:
    """Rata-rata SSIM atas batch. x, y: (B, 1, H, W) dalam [0, 1]."""
    xs = x.detach().cpu().numpy()
    ys = y.detach().cpu().numpy()
    vals = [structural_similarity(a[0], b[0], data_range=1.0) for a, b in zip(xs, ys)]
    return float(sum(vals) / len(vals))


@torch.no_grad()
def evaluate_reconstruction(recon_fn, loader, device) -> dict:
    """recon_fn(x) -> x_hat di [0, 1]. Mengembalikan rata-rata MSE/PSNR/SSIM atas loader."""
    n, mse_s, psnr_s, ssim_s = 0, 0.0, 0.0, 0.0
    for x, _ in loader:
        x = x.to(device)
        x_hat = recon_fn(x).clamp(0, 1)
        b = x.size(0)
        mse_s += mse_per_image(x, x_hat).sum().item()
        psnr_s += psnr_per_image(x, x_hat).sum().item()
        ssim_s += ssim_batch(x, x_hat) * b
        n += b
    return {"mse": mse_s / n, "psnr": psnr_s / n, "ssim": ssim_s / n}
