"""Helper visualisasi grid citra (dipakai semua figur)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402


def minmax_per_image(x: torch.Tensor) -> torch.Tensor:
    """Normalisasi tiap citra ke [0, 1] (untuk menampilkan hasil proyeksi balik)."""
    flat = x.flatten(1)
    lo = flat.min(1, keepdim=True).values
    hi = flat.max(1, keepdim=True).values
    return ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(x)


def show_grid(ax, imgs: torch.Tensor, nrow: int, title: str = "", padding: int = 2):
    """imgs: (N, 1, 28, 28) dalam [0, 1]."""
    grid = make_grid(imgs.detach().cpu().clamp(0, 1), nrow=nrow, padding=padding, pad_value=1.0)
    ax.imshow(grid[0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save_grid(imgs: torch.Tensor, nrow: int, path, title: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (imgs.size(0) + nrow - 1) // nrow
    fig, ax = plt.subplots(figsize=(nrow * 0.9, rows * 0.9 + 0.4))
    show_grid(ax, imgs, nrow, title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
