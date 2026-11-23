"""Latih Vanilla GAN (MLP) dan DCGAN pada Fashion-MNIST.

Contoh:
    python train_gan.py --model vanilla_gan
    python train_gan.py --model dcgan
    python train_gan.py --model all --epochs 30

Yang dicatat untuk analisis kestabilan (Q3):
  - loss D dan G per epoch, rata-rata D(x) dan D(G(z))
  - grid sampel dari noise tetap tiap N epoch (memantau mode collapse secara visual)
  - cakupan kelas (mode coverage): 10.000 sampel diklasifikasi oleh MLP Classifier
    Bagian B (jalankan experiment_mlp.py dulu); entropi tinggi = beragam,
    entropi rendah = indikasi mode collapse
  - waktu per epoch dan latensi generasi 64 sampel
"""
import argparse
import math
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import CKPT_DIR, GANConfig, RESULTS_DIR
from src.data import CLASS_NAMES, get_loaders, to_minus1_1, to_zero1
from src.models.gan import build_gan
from src.models.mlp import MLPClassifier
from src.utils import get_device, measure_latency, save_checkpoint, save_json, set_seed
from src.viz import save_grid

MODELS = ("vanilla_gan", "dcgan")
FIG_DIR = RESULTS_DIR / "figures"


def load_classifier(device):
    path = CKPT_DIR / "mlp_classifier.pt"
    if not path.exists():
        return None
    clf = MLPClassifier().to(device)
    clf.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    return clf.eval()


@torch.no_grad()
def mode_coverage(G, clf, z_dim, device, n=10000, bs=500):
    """Distribusi kelas sampel G menurut MLP Classifier."""
    counts = torch.zeros(10)
    for i in range(0, n, bs):
        x = to_zero1(G(torch.randn(bs, z_dim, device=device)))
        counts += torch.bincount(clf(x).argmax(1).cpu(), minlength=10).float()
    p = counts / counts.sum()
    ent = max(0.0, -(p[p > 0] * p[p > 0].log()).sum().item() / math.log(10))   # 1.0 = seragam
    return {"class_fraction": {CLASS_NAMES[c]: round(p[c].item(), 4) for c in range(10)},
            "normalized_entropy": ent,
            "classes_over_2pct": int((p > 0.02).sum().item())}


def train_one(name, cfg: GANConfig, epochs, device, max_batches=0):
    set_seed()
    train_loader, _, _ = get_loaders()
    G, D = build_gan(name, cfg.z_dim)
    G, D = G.to(device), D.to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    fixed_z = torch.randn(16, cfg.z_dim, device=device, generator=None)

    hist = {k: [] for k in ("d_loss", "g_loss", "d_real", "d_fake", "epoch_time_s")}
    for ep in range(1, epochs + 1):
        G.train()
        D.train()
        t0 = time.perf_counter()
        sums = {k: 0.0 for k in ("d_loss", "g_loss", "d_real", "d_fake")}
        nb = 0
        for i, (x, _) in enumerate(tqdm(train_loader, desc=f"{name} ep{ep}/{epochs}", leave=False)):
            if max_batches and i >= max_batches:
                break
            x = to_minus1_1(x.to(device))
            b = x.size(0)
            ones = torch.ones(b, device=device)
            zeros = torch.zeros(b, device=device)

            # --- Diskriminator: max log D(x) + log(1 - D(G(z)))
            z = torch.randn(b, cfg.z_dim, device=device)
            fake = G(z)
            logit_real = D(x)
            logit_fake = D(fake.detach())
            d_loss = (F.binary_cross_entropy_with_logits(logit_real, ones)
                      + F.binary_cross_entropy_with_logits(logit_fake, zeros))
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # --- Generator: loss non-saturating, max log D(G(z))
            logit_gen = D(fake)
            g_loss = F.binary_cross_entropy_with_logits(logit_gen, ones)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            sums["d_loss"] += d_loss.item()
            sums["g_loss"] += g_loss.item()
            sums["d_real"] += torch.sigmoid(logit_real).mean().item()
            sums["d_fake"] += torch.sigmoid(logit_fake).mean().item()
            nb += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        for k in sums:
            hist[k].append(sums[k] / nb)
        hist["epoch_time_s"].append(dt)
        print(f"[{name}] epoch {ep:3d}  D {hist['d_loss'][-1]:.4f}  G {hist['g_loss'][-1]:.4f}  "
              f"D(x) {hist['d_real'][-1]:.3f}  D(G(z)) {hist['d_fake'][-1]:.3f}  ({dt:.1f}s)")

        if ep == 1 or ep % cfg.progress_every == 0 or ep == epochs:
            G.eval()
            with torch.no_grad():
                save_grid(to_zero1(G(fixed_z)), 4, FIG_DIR / "gan_progress" / f"{name}_ep{ep:03d}.png",
                          f"{name} epoch {ep}")

    G.eval()
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(42)
        z16 = torch.randn(16, cfg.z_dim, generator=g).to(device)
        save_grid(to_zero1(G(z16)), 4, FIG_DIR / f"samples_{name}.png", name)

    latency = measure_latency(lambda: G(torch.randn(64, cfg.z_dim, device=device)))
    clf = load_classifier(device)
    coverage = mode_coverage(G, clf, cfg.z_dim, device) if clf is not None else None
    if coverage is None:
        print("PERINGATAN: checkpoints/mlp_classifier.pt tidak ada; mode coverage dilewati.")
    else:
        print(f"[{name}] mode coverage: entropi {coverage['normalized_entropy']:.3f}, "
              f"kelas >2%: {coverage['classes_over_2pct']}/10")

    result = {
        "model": name,
        "epochs": epochs,
        "z_dim": cfg.z_dim,
        "n_params_G": sum(p.numel() for p in G.parameters()),
        "n_params_D": sum(p.numel() for p in D.parameters()),
        "avg_epoch_time_s": sum(hist["epoch_time_s"]) / len(hist["epoch_time_s"]),
        "latency_ms_per_batch64": latency,
        "mode_coverage": coverage,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "history": hist,
    }
    save_checkpoint({"model": name, "G": G.state_dict(), "D": D.state_dict(), "config": cfg.__dict__},
                    CKPT_DIR / f"{name}.pt")
    save_json(result, RESULTS_DIR / f"{name}.json")
    print(f"[{name}] selesai. latensi {latency:.2f} ms / 64 sampel")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS + ("all",), default="all")
    p.add_argument("--epochs", type=int, default=GANConfig.epochs)
    p.add_argument("--max-batches", type=int, default=0, help="batasi batch/epoch (smoke test)")
    args = p.parse_args()

    device = get_device()
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    cfg = GANConfig()
    for n in (MODELS if args.model == "all" else (args.model,)):
        train_one(n, cfg, args.epochs, device, args.max_batches)


if __name__ == "__main__":
    main()
