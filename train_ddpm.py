"""Latih DDPM pada Fashion-MNIST (model terberat; bisa dilanjutkan dengan --resume).

Contoh:
    python train_ddpm.py --epochs 30
    python train_ddpm.py --epochs 30 --resume      # lanjutkan jika training terputus

Output:
    checkpoints/ddpm_last.pt         checkpoint tiap epoch (untuk resume)
    checkpoints/ddpm.pt              bobot EMA final
    results/ddpm.json                loss per epoch, waktu/epoch, latensi sampling, mode coverage
    results/figures/samples_ddpm.png 4x4 sampel (Figur 2)
    results/figures/ddpm_trajectory.png  x_T -> x_0 (Figur 5)
    results/figures/gan_progress/ddpm_epXXX.png  sampel per beberapa epoch
"""
import argparse
import copy
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import CKPT_DIR, RESULTS_DIR  # noqa: E402
from src.data import CLASS_NAMES, get_loaders, to_minus1_1, to_zero1  # noqa: E402
from src.models.ddpm import EMA, GaussianDiffusion, UNet  # noqa: E402
from src.models.mlp import MLPClassifier  # noqa: E402
from src.utils import get_device, save_checkpoint, save_json, set_seed  # noqa: E402
from src.viz import save_grid, show_grid  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
TRAJ_STEPS = [999, 900, 700, 500, 300, 200, 100, 50, 0]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def build(args, device):
    unet = UNet(args.ch)
    diff = GaussianDiffusion(unet, T=args.T).to(device)
    return diff


def sample_grid(diff_ema, n, device, seed=42):
    g = torch.Generator().manual_seed(seed)
    return to_zero1(diff_ema.sample(n, device, generator=g))


@torch.no_grad()
def mode_coverage(diff_ema, device, n, bs):
    path = CKPT_DIR / "mlp_classifier.pt"
    if not path.exists():
        print("PERINGATAN: mlp_classifier.pt tidak ada; mode coverage dilewati.")
        return None
    clf = MLPClassifier().to(device)
    clf.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    clf.eval()
    counts = torch.zeros(10)
    for i in tqdm(range(0, n, bs), desc="coverage", leave=False):
        x = to_zero1(diff_ema.sample(min(bs, n - i), device))
        counts += torch.bincount(clf(x).argmax(1).cpu(), minlength=10).float()
    p = counts / counts.sum()
    ent = max(0.0, -(p[p > 0] * p[p > 0].log()).sum().item() / math.log(10))
    return {"n_samples": n,
            "class_fraction": {CLASS_NAMES[c]: round(p[c].item(), 4) for c in range(10)},
            "normalized_entropy": ent,
            "classes_over_2pct": int((p > 0.02).sum().item())}


def save_trajectory(traj, path):
    """Baris = sampel, kolom = langkah t (x_T -> x_0)."""
    keys = [t for t in TRAJ_STEPS if t in traj] + [-1]
    n = traj[keys[0]].size(0)
    cols = [to_zero1(traj[k].clamp(-1, 1)) for k in keys]
    imgs = torch.stack(cols, 1).reshape(n * len(keys), 1, 28, 28)
    fig, ax = plt.subplots(figsize=(len(keys) * 0.9, n * 0.9 + 0.6))
    show_grid(ax, imgs, len(keys))
    labels = ["x_T" if k == 999 else ("x_0" if k == -1 else f"t={k}") for k in keys]
    ax.set_title("Trajektori denoising DDPM:  " + "  →  ".join(labels), fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--ch", type=int, default=64, help="channel dasar U-Net")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--coverage-n", type=int, default=1000, help="jumlah sampel untuk mode coverage")
    p.add_argument("--sample-bs", type=int, default=250, help="batch saat sampling massal")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-batches", type=int, default=0, help="batasi batch/epoch (smoke test)")
    args = p.parse_args()

    device = get_device()
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    set_seed()
    train_loader, val_loader, _ = get_loaders(args.batch_size)

    diff = build(args, device)
    opt = torch.optim.Adam(diff.model.parameters(), lr=args.lr)
    ema = EMA(diff.model, args.ema)
    hist = {"train_loss": [], "val_loss": [], "epoch_time_s": []}
    start_ep = 1
    last_path = CKPT_DIR / "ddpm_last.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        diff.model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        hist = ck["history"]
        start_ep = ck["epoch"] + 1
        torch.set_rng_state(ck["rng_cpu"])
        if device.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"])
        print(f"Melanjutkan dari epoch {start_ep}")

    n_params = sum(p.numel() for p in diff.model.parameters())
    print(f"U-Net: {n_params / 1e6:.2f} juta parameter, T = {args.T}")

    ema_model = copy.deepcopy(diff)   # wadah bobot EMA untuk sampling

    for ep in range(start_ep, args.epochs + 1):
        diff.train()
        t0 = time.perf_counter()
        run, n = 0.0, 0
        for i, (x, _) in enumerate(tqdm(train_loader, desc=f"ddpm ep{ep}/{args.epochs}", leave=False)):
            if args.max_batches and i >= args.max_batches:
                break
            x = to_minus1_1(x.to(device))
            loss = diff.loss(x)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff.model.parameters(), 1.0)
            opt.step()
            ema.update(diff.model)
            run += loss.item() * x.size(0)
            n += x.size(0)
        sync(device)
        dt = time.perf_counter() - t0

        diff.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for j, (x, _) in enumerate(val_loader):
                if args.max_batches and j >= args.max_batches:
                    break
                x = to_minus1_1(x.to(device))
                vl += diff.loss(x).item() * x.size(0)
                vn += x.size(0)
        hist["train_loss"].append(run / n)
        hist["val_loss"].append(vl / vn)
        hist["epoch_time_s"].append(dt)
        print(f"[ddpm] epoch {ep:3d}  train {run / n:.5f}  val {vl / vn:.5f}  ({dt:.1f}s)")

        save_checkpoint({"epoch": ep, "model": diff.model.state_dict(), "opt": opt.state_dict(),
                         "ema": ema.shadow, "history": hist, "args": vars(args),
                         "rng_cpu": torch.get_rng_state(),
                         "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None},
                        last_path)

        if ep == 1 or ep % args.progress_every == 0 or ep == args.epochs:
            ema.copy_to(ema_model.model)
            ema_model.eval()
            save_grid(sample_grid(ema_model, 16, device), 4,
                      FIG_DIR / "gan_progress" / f"ddpm_ep{ep:03d}.png", f"ddpm (EMA) epoch {ep}")

    # ---------------------------------------------------------- evaluasi final (bobot EMA)
    ema.copy_to(ema_model.model)
    ema_model.eval()
    save_checkpoint({"model": "ddpm", "state_dict": ema_model.model.state_dict(), "args": vars(args)},
                    CKPT_DIR / "ddpm.pt")
    save_grid(sample_grid(ema_model, 16, device), 4, FIG_DIR / "samples_ddpm.png", "ddpm")

    g = torch.Generator().manual_seed(7)
    _, traj = ema_model.sample(6, device, record_steps=set(TRAJ_STEPS), generator=g)
    save_trajectory(traj, FIG_DIR / "ddpm_trajectory.png")

    # latensi: 64 sampel, T langkah penuh (diukur 3x, setelah 1x pemanasan singkat)
    with torch.no_grad():
        xw = torch.randn(64, 1, 28, 28, device=device)
        for t in range(10):
            ema_model.p_sample(xw, args.T - 1 - t)
    times = []
    for _ in range(3):
        sync(device)
        t0 = time.perf_counter()
        ema_model.sample(64, device)
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    latency = sum(times) / len(times)
    print(f"[ddpm] latensi {latency:.0f} ms / 64 sampel ({args.T} langkah)")

    coverage = mode_coverage(ema_model, device, args.coverage_n, args.sample_bs)
    if coverage:
        print(f"[ddpm] mode coverage: entropi {coverage['normalized_entropy']:.3f}, "
              f"kelas >2%: {coverage['classes_over_2pct']}/10")

    save_json({
        "model": "ddpm",
        "epochs": len(hist["train_loss"]),
        "T": args.T,
        "n_params": n_params,
        "avg_epoch_time_s": sum(hist["epoch_time_s"]) / len(hist["epoch_time_s"]),
        "latency_ms_per_batch64": latency,
        "latency_runs_ms": times,
        "mode_coverage": coverage,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "history": hist,
    }, RESULTS_DIR / "ddpm.json")
    print("[ddpm] selesai.")


if __name__ == "__main__":
    main()
