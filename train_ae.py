"""Latih Vanilla AE / Sparse AE / VAE pada Fashion-MNIST.

Contoh:
    python train_ae.py --model vanilla_ae
    python train_ae.py --model sparse_ae
    python train_ae.py --model vae
    python train_ae.py --model all --epochs 30
"""
import argparse
import time

import torch
from tqdm import tqdm

from src.config import AEConfig, CKPT_DIR, LATENT_DIM, RESULTS_DIR
from src.data import get_loaders
from src.metrics import evaluate_reconstruction
from src.models.autoencoder import SparseAE, VAE, VanillaAE
from src.utils import get_device, measure_latency, save_checkpoint, save_json, set_seed

MODELS = ("vanilla_ae", "sparse_ae", "vae")


def build(name: str, cfg: AEConfig):
    if name == "vanilla_ae":
        return VanillaAE(LATENT_DIM)
    if name == "sparse_ae":
        return SparseAE(LATENT_DIM, cfg.sparse_hidden, cfg.sparse_lambda)
    if name == "vae":
        return VAE(LATENT_DIM, cfg.vae_beta)
    raise ValueError(name)


def recon_fn_for(model):
    if isinstance(model, VAE):
        return model.reconstruct   # pakai mean (deterministik) saat evaluasi
    return model


@torch.no_grad()
def val_loss(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        l, _ = model.loss(x)
        tot += l.item() * x.size(0)
        n += x.size(0)
    return tot / n


def train_one(name: str, cfg: AEConfig, epochs: int, device, num_workers: int, max_batches: int = 0):
    set_seed()
    train_loader, val_loader, test_loader = get_loaders(num_workers=num_workers)
    model = build(name, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"train_loss": [], "val_loss": [], "epoch_time_s": []}
    best_val, best_state = float("inf"), None

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        run, n = 0.0, 0
        for i, (x, _) in enumerate(tqdm(train_loader, desc=f"{name} ep{ep}/{epochs}", leave=False)):
            if max_batches and i >= max_batches:
                break
            x = x.to(device)
            opt.zero_grad()
            loss, _ = model.loss(x)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            n += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        tr = run / n
        vl = val_loss(model, val_loader, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["epoch_time_s"].append(dt)
        print(f"[{name}] epoch {ep:3d}  train {tr:.5f}  val {vl:.5f}  ({dt:.1f}s)")

        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # --- evaluasi test set (MSE/PSNR/SSIM pada [0,1]) ---
    metrics = evaluate_reconstruction(recon_fn_for(model), test_loader, device)

    # --- latensi inferensi per batch 64 ---
    x64 = next(iter(test_loader))[0][:64].to(device)
    if isinstance(model, VAE):
        latency = measure_latency(lambda: model.sample(64, device))   # generasi dari prior
        latency_recon = measure_latency(lambda: model.reconstruct(x64))
    else:
        latency = measure_latency(lambda: model(x64))                 # rekonstruksi
        latency_recon = latency

    result = {
        "model": name,
        "latent_dim": LATENT_DIM,
        "epochs": epochs,
        "avg_epoch_time_s": sum(history["epoch_time_s"]) / len(history["epoch_time_s"]),
        "latency_ms_per_batch64": latency,
        "latency_recon_ms_per_batch64": latency_recon,
        **{f"test_{k}": v for k, v in metrics.items()},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "history": history,
    }
    print(f"[{name}] TEST  MSE {metrics['mse']:.5f}  PSNR {metrics['psnr']:.2f} dB  SSIM {metrics['ssim']:.4f}")

    save_checkpoint({"model": name, "state_dict": best_state, "config": cfg.__dict__}, CKPT_DIR / f"{name}.pt")
    save_json(result, RESULTS_DIR / f"{name}.json")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS + ("all",), default="all")
    p.add_argument("--epochs", type=int, default=AEConfig.epochs)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=0, help="batasi batch/epoch (smoke test)")
    args = p.parse_args()

    device = get_device()
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    cfg = AEConfig()
    names = MODELS if args.model == "all" else (args.model,)
    for n in names:
        train_one(n, cfg, args.epochs, device, args.num_workers, args.max_batches)


if __name__ == "__main__":
    main()
