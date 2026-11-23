"""Bagian B: Membuktikan MLP Classifier tidak bisa dipakai untuk image generation.

Tahap:
  1. Latih MLP 784->256->64->10 (Cross-Entropy) sampai test acc >= 88%.
     Simpan W1, W2, W3 dan z_MLP (64-d).
  2. Proyeksi balik dari one-hot label:  x = W1^T W2^T W3^T e_c
  3. Proyeksi balik dari z_MLP gambar uji: x = W1^T W2^T z
  4. Frozen latent decoder 64->256->784 (MSE) dari z_MLP; bandingkan dgn Vanilla AE.
  5. Sampling acak z ~ N(mu_z, diag(sigma_z^2)) -> decoder tahap 4.
Output: results/mlp_experiment.json dan results/figures/fig3_mlp_proof.png (+ panel terpisah).

Jalankan train_ae.py terlebih dahulu agar perbandingan dengan Vanilla AE tersedia.
    python experiment_mlp.py
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import CKPT_DIR, LATENT_DIM, RESULTS_DIR  # noqa: E402
from src.data import CLASS_NAMES, get_loaders  # noqa: E402
from src.metrics import evaluate_reconstruction, mse_per_image, psnr_per_image, ssim_batch  # noqa: E402
from src.models.autoencoder import Decoder, VanillaAE  # noqa: E402
from src.models.mlp import MLPClassifier  # noqa: E402
from src.utils import get_device, save_checkpoint, save_json, set_seed  # noqa: E402
from src.viz import minmax_per_image, save_grid, show_grid  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
TARGET_ACC = 0.88


# ---------------------------------------------------------------- Tahap 1
@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        n += y.numel()
    return correct / n


def stage1_train_classifier(loaders, device, epochs, lr):
    train_loader, val_loader, test_loader = loaders
    set_seed()
    model = MLPClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"train_loss": [], "val_acc": []}
    best_acc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        run, n = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"mlp ep{ep}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * y.numel()
            n += y.numel()
        va = accuracy(model, val_loader, device)
        hist["train_loss"].append(run / n)
        hist["val_acc"].append(va)
        print(f"[mlp] epoch {ep:3d}  train CE {run / n:.4f}  val acc {va:.4f}")
        if va > best_acc:
            best_acc = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_acc = accuracy(model, test_loader, device)
    status = "OK" if test_acc >= TARGET_ACC else "DI BAWAH TARGET 88% (tambah --mlp-epochs)"
    print(f"[mlp] TEST accuracy {test_acc:.4f}  -> {status}")

    W1, W2, W3 = model.weights()
    save_checkpoint({"state_dict": best_state}, CKPT_DIR / "mlp_classifier.pt")
    save_checkpoint({"W1": W1.cpu(), "W2": W2.cpu(), "W3": W3.cpu()}, CKPT_DIR / "mlp_weights.pt")
    model.eval()
    return model, test_acc, hist


@torch.no_grad()
def collect_z(model, loader, device):
    zs, xs, ys = [], [], []
    for x, y in loader:
        zs.append(model.features(x.to(device)).cpu())
        xs.append(x)
        ys.append(y)
    return torch.cat(zs), torch.cat(xs), torch.cat(ys)


def pick_one_per_class(y, n_classes=8):
    """Indeks gambar uji pertama untuk kelas 0..n_classes-1 (contoh yang konsisten)."""
    return torch.stack([(y == c).nonzero()[0, 0] for c in range(n_classes)])


def transpose_report(W1, W2, W3):
    """Seberapa jauh W W^T dari identitas (jika W ortogonal, transpos = invers)."""
    out = {}
    for name, W in (("W1", W1), ("W2", W2), ("W3", W3)):
        G = W @ W.T                      # (out x out)
        I = torch.eye(G.size(0), device=G.device)
        out[name] = {"shape": list(W.shape),
                     "rel_fro_WWt_minus_I": ((G - I).norm() / I.norm()).item(),
                     "rank": int(torch.linalg.matrix_rank(W).item())}
    M = W3 @ W2 @ W1                     # (10 x 784): seluruh jaringan tanpa ReLU
    out["W3W2W1_rank"] = int(torch.linalg.matrix_rank(M).item())
    return out


# ---------------------------------------------------------------- Tahap 4
def stage4_frozen_decoder(z_tr, x_tr, z_va, x_va, device, epochs, lr, batch_size):
    set_seed()
    dec = Decoder(LATENT_DIM).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    n = z_tr.size(0)
    hist = {"train_loss": [], "val_loss": []}
    best, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        dec.train()
        perm = torch.randperm(n)
        run = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            z, x = z_tr[idx].to(device), x_tr[idx].to(device)
            opt.zero_grad()
            loss = F.mse_loss(dec(z), x)
            loss.backward()
            opt.step()
            run += loss.item() * idx.numel()
        with torch.no_grad():
            dec.eval()
            vl = F.mse_loss(dec(z_va.to(device)), x_va.to(device)).item()
        hist["train_loss"].append(run / n)
        hist["val_loss"].append(vl)
        print(f"[frozen-dec] epoch {ep:3d}  train {run / n:.5f}  val {vl:.5f}")
        if vl < best:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}
    dec.load_state_dict(best_state)
    dec.eval()
    save_checkpoint({"state_dict": best_state}, CKPT_DIR / "mlp_frozen_decoder.pt")
    return dec, hist


def load_vanilla_ae(device):
    path = CKPT_DIR / "vanilla_ae.pt"
    if not path.exists():
        return None
    ae = VanillaAE(LATENT_DIM).to(device)
    ae.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    return ae.eval()


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp-epochs", type=int, default=20)
    p.add_argument("--dec-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    device = get_device()
    print("Device:", device)
    loaders = get_loaders(args.batch_size)
    _, _, test_loader = loaders

    # 1. Classifier
    model, test_acc, mlp_hist = stage1_train_classifier(loaders, device, args.mlp_epochs, args.lr)
    W1, W2, W3 = (w.to(device) for w in model.weights())

    z_tr, x_tr, _ = collect_z(model, loaders[0], device)
    z_va, x_va, _ = collect_z(model, loaders[1], device)
    z_te, x_te, y_te = collect_z(model, test_loader, device)
    save_checkpoint({"z_train": z_tr, "z_test": z_te, "y_test": y_te}, CKPT_DIR / "z_mlp.pt")
    dead = (z_tr.max(0).values == 0).sum().item()     # dimensi z yang selalu 0 (ReLU mati)
    sparsity = (z_tr == 0).float().mean().item()       # proporsi aktivasi nol

    with torch.no_grad():
        # 2. One-hot -> piksel lewat transpos bobot (tanpa bias/ReLU)
        E = torch.eye(10, device=device)
        x_onehot = (E @ W3 @ W2 @ W1).view(10, 1, 28, 28)
        class_means = torch.stack([x_te[y_te == c].mean(0) for c in range(10)]).to(device)

        # 3. z_MLP gambar uji -> piksel lewat transpos bobot
        idx = pick_one_per_class(y_te, 8)
        x_orig = x_te[idx].to(device)
        z_sel = z_te[idx].to(device)
        x_bp = (z_sel @ W2 @ W1).view(-1, 1, 28, 28)
        x_bp_n = minmax_per_image(x_bp)
        bp_metrics = {"mse": mse_per_image(x_orig, x_bp_n).mean().item(),
                      "psnr": psnr_per_image(x_orig, x_bp_n).mean().item(),
                      "ssim": ssim_batch(x_orig, x_bp_n),
                      "note": "x_bp dinormalisasi min-max per citra sebelum dibandingkan"}

    # 4. Frozen latent decoder
    dec, dec_hist = stage4_frozen_decoder(z_tr, x_tr, z_va, x_va, device,
                                          args.dec_epochs, args.lr, args.batch_size)
    frozen_metrics = evaluate_reconstruction(lambda x: dec(model.features(x)), test_loader, device)
    print(f"[frozen-dec] TEST  MSE {frozen_metrics['mse']:.5f}  PSNR {frozen_metrics['psnr']:.2f}  "
          f"SSIM {frozen_metrics['ssim']:.4f}")

    ae = load_vanilla_ae(device)
    ae_metrics = None
    ae_json = RESULTS_DIR / "vanilla_ae.json"
    if ae_json.exists():
        r = json.loads(ae_json.read_text())
        ae_metrics = {k: r[f"test_{k}"] for k in ("mse", "psnr", "ssim")}
        print(f"[vanilla_ae] TEST MSE {ae_metrics['mse']:.5f}  PSNR {ae_metrics['psnr']:.2f}  "
              f"SSIM {ae_metrics['ssim']:.4f}")
    else:
        print("PERINGATAN: results/vanilla_ae.json belum ada; jalankan train_ae.py dulu.")

    # 5. Sampling acak dari statistik z_MLP
    mu, sigma = z_tr.mean(0), z_tr.std(0)
    g = torch.Generator().manual_seed(42)
    z_rand = mu + sigma * torch.randn(16, LATENT_DIM, generator=g)
    with torch.no_grad():
        x_rand = dec(z_rand.to(device))
        x_dec = dec(z_sel)
        x_ae = ae(x_orig) if ae is not None else None

    # ---------------------------------------------------------- figur
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_grid(minmax_per_image(x_onehot), 10, FIG_DIR / "fig3a_onehot_backproj.png",
              "Proyeksi balik one-hot (W1ᵀW2ᵀW3ᵀ e_c)")
    save_grid(torch.cat([x_orig, x_bp_n]), 8, FIG_DIR / "fig3b_zmlp_backproj.png",
              "Atas: asli | Bawah: W1ᵀW2ᵀ z_MLP")
    rows_c = [x_orig, x_dec] + ([x_ae] if x_ae is not None else [])
    save_grid(torch.cat(rows_c), 8, FIG_DIR / "fig3c_frozen_decoder.png",
              "Asli | Frozen decoder" + (" | Vanilla AE" if x_ae is not None else ""))
    save_grid(x_rand, 4, FIG_DIR / "fig3d_random_samples.png", "z ~ N(μ_z, diag σ_z²)")

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])
    ax = fig.add_subplot(gs[0, :])
    show_grid(ax, torch.cat([minmax_per_image(x_onehot), class_means]), 10,
              "(a) Atas: proyeksi balik one-hot W1ᵀW2ᵀW3ᵀ·e_c per kelas | Bawah: rata-rata citra kelas")
    ax = fig.add_subplot(gs[1, 0])
    show_grid(ax, torch.cat([x_orig, x_bp_n] + rows_c[1:]), 8,
              "(b, c) Asli | W1ᵀW2ᵀ·z_MLP | Frozen decoder" + (" | Vanilla AE" if x_ae is not None else ""))
    ax = fig.add_subplot(gs[1, 1])
    show_grid(ax, x_rand, 4, "(d) Decoder(z), z ~ N(μ_z, diag σ_z²)")
    fig.suptitle("Figur 3. Bukti MLP Classifier tidak dapat menghasilkan citra", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_mlp_proof.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    result = {
        "mlp_test_accuracy": test_acc,
        "mlp_target_met": test_acc >= TARGET_ACC,
        "z_mlp": {"dim": LATENT_DIM, "dead_units": dead, "zero_fraction": sparsity},
        "transpose_vs_inverse": transpose_report(W1, W2, W3),
        "backproj_from_zmlp": bp_metrics,
        "frozen_decoder_test": frozen_metrics,
        "vanilla_ae_test": ae_metrics,
        "test_image_classes": [CLASS_NAMES[c] for c in range(8)],
        "history": {"mlp": mlp_hist, "frozen_decoder": dec_hist},
    }
    save_json(result, RESULTS_DIR / "mlp_experiment.json")
    print("Selesai. Figur di", FIG_DIR, "| ringkasan di results/mlp_experiment.json")


if __name__ == "__main__":
    main()
