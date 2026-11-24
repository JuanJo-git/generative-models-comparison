"""Evaluasi gabungan semua model (tanpa training ulang; memakai checkpoint).

Menghasilkan:
  results/eval.json              FID, statistik sparsity, jarak laten saat interpolasi
  results/metrics_table.md/.csv  tabel metrik lengkap (template laporan hal. 4)
  results/figures/fig1_reconstruction.png    asli | Vanilla AE | Sparse AE | VAE
  results/figures/fig2_samples.png           4x4 sampel VAE, Vanilla GAN, DCGAN, DDPM
  results/figures/fig4_interpolation.png     interpolasi laten 5 model (alpha 0..1)
  results/figures/fig4b_latent_distance.png  jarak z(alpha) ke kode data terdekat (AE vs VAE)
  results/figures/fig5_training.png          kurva training 6 model + trajektori DDPM
  results/figures/figx_sparse_features.png   fitur encoder Vanilla AE vs Sparse AE (Q2)

Protokol FID (sama untuk semua model):
  Inception-v3 pool3 (2048-d) dari torchmetrics; citra 28x28 grayscale diulang ke 3 kanal,
  di-resize internal ke 299x299. Distribusi acuan = seluruh test set (7.000 citra).
  Tiap model: N sampel (default 5.000) dari noise acak / prior, seed tetap.
  Baseline: 5.000 citra train asli vs test set (batas bawah FID yang realistis).

Sampel disimpan di checkpoints/ (cache), jadi DDPM yang lambat cukup di-sample sekali.
    python evaluate.py                 # semua
    python evaluate.py --n-fid 2000    # lebih cepat (angka FID lebih bias ke atas)
    python evaluate.py --skip-fid      # hanya tabel (pakai FID lama jika ada) & figur
"""
import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import AEConfig, CKPT_DIR, LATENT_DIM, RESULTS_DIR, SEED, Z_DIM_GAN  # noqa: E402
from src.data import CLASS_NAMES, get_loaders, to_zero1  # noqa: E402
from src.models.autoencoder import SparseAE, VAE, VanillaAE  # noqa: E402
from src.models.ddpm import GaussianDiffusion, UNet  # noqa: E402
from src.models.gan import build_gan  # noqa: E402
from src.utils import get_device, save_json  # noqa: E402
from src.viz import show_grid  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # palet kategorikal tervalidasi (slot 1, 2)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
GEN_MODELS = ("vae", "vanilla_gan", "dcgan", "ddpm")
AE_MODELS = ("vanilla_ae", "sparse_ae", "vae")
LABEL = {"vanilla_ae": "Vanilla AE", "sparse_ae": "Sparse AE", "vae": "VAE",
         "vanilla_gan": "Vanilla GAN", "dcgan": "DCGAN", "ddpm": "DDPM",
         "mlp_frozen_decoder": "MLP + frozen decoder", "real_train": "Data asli (train)"}


# ------------------------------------------------------------------ loading
def load_models(device):
    cfg = AEConfig()
    m = {}
    for name, ctor in (("vanilla_ae", lambda: VanillaAE(LATENT_DIM)),
                       ("sparse_ae", lambda: SparseAE(LATENT_DIM, cfg.sparse_hidden, cfg.sparse_lambda)),
                       ("vae", lambda: VAE(LATENT_DIM, cfg.vae_beta))):
        net = ctor().to(device)
        net.load_state_dict(torch.load(CKPT_DIR / f"{name}.pt", map_location=device)["state_dict"])
        m[name] = net.eval()
    for name in ("vanilla_gan", "dcgan"):
        G, _ = build_gan(name, Z_DIM_GAN)
        G.load_state_dict(torch.load(CKPT_DIR / f"{name}.pt", map_location=device)["G"])
        m[name] = G.to(device).eval()
    ck = torch.load(CKPT_DIR / "ddpm.pt", map_location=device)
    diff = GaussianDiffusion(UNet(ck["args"]["ch"]), T=ck["args"]["T"]).to(device)
    diff.model.load_state_dict(ck["state_dict"])
    m["ddpm"] = diff.eval()
    return m


def n_params(net):
    return sum(p.numel() for p in net.parameters())


# ------------------------------------------------------------------ sampling (cache)
@torch.no_grad()
def generate(name, model, n, device, bs):
    """n sampel di [0, 1], (n, 1, 28, 28), CPU. Di-cache ke checkpoints/."""
    path = CKPT_DIR / f"samples_{name}_{n}.pt"
    if path.exists():
        return torch.load(path)
    torch.manual_seed(SEED + 1)
    out = []
    bs_eff = bs if name == "ddpm" else 1000
    for i in tqdm(range(0, n, bs_eff), desc=f"sampling {name}"):
        b = min(bs_eff, n - i)
        if name == "vae":
            x = model.sample(b, device)
        elif name == "ddpm":
            x = to_zero1(model.sample(b, device))
        else:
            x = to_zero1(model(torch.randn(b, Z_DIM_GAN, device=device)))
        out.append(x.clamp(0, 1).cpu())
    x = torch.cat(out)
    torch.save(x, path)
    return x


def to_fid_input(x):
    return x.repeat(1, 3, 1, 1)


def compute_fid(samples, x_test, x_train_ref, device, bs=200):
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
    for i in tqdm(range(0, len(x_test), bs), desc="FID real (test)"):
        fid.update(to_fid_input(x_test[i:i + bs]).to(device), real=True)
    scores = {}
    for name, xs in [("real_train", x_train_ref)] + list(samples.items()):
        fid.reset()   # hanya menghapus statistik fake
        for i in range(0, len(xs), bs):
            fid.update(to_fid_input(xs[i:i + bs]).to(device), real=False)
        scores[name] = fid.compute().item()
        print(f"FID {LABEL[name]:<20} {scores[name]:8.2f}")
    return scores


# ------------------------------------------------------------------ analisis tambahan
@torch.no_grad()
def sparsity_stats(models, x_test, device):
    """Aktivasi lapisan tersembunyi 256-ReLU encoder: Vanilla AE vs Sparse AE (Q1/Q2)."""
    out = {}
    hooks = {"vanilla_ae": lambda m, x: torch.relu(m.encoder.net[1](x.flatten(1))),
             "sparse_ae": lambda m, x: m.enc_sparse(x)}
    for name, fn in hooks.items():
        h = torch.cat([fn(models[name], x_test[i:i + 1000].to(device)).cpu()
                       for i in range(0, len(x_test), 1000)])
        active = (h > 1e-6).float()
        out[name] = {"mean_abs_activation": h.abs().mean().item(),
                     "fraction_active": active.mean().item(),
                     "mean_active_units_per_image": active.sum(1).mean().item(),
                     "dead_units": int((active.sum(0) == 0).sum().item()),
                     "hidden_units": h.size(1)}
    return out


@torch.no_grad()
def encode_all(model, x, device):
    return torch.cat([model.encode(x[i:i + 1000].to(device)).cpu() for i in range(0, len(x), 1000)])


@torch.no_grad()
def latent_gap(models, x_train, xa, xb, device, alphas):
    """Jarak z(alpha) ke kode data latih terdekat, dinormalisasi median jarak NN antar-kode.
    Nilai >> 1 di tengah lintasan = z berada di wilayah kosong ruang laten (Q4)."""
    res = {}
    for name in AE_MODELS:
        m = models[name]
        Z = encode_all(m, x_train, device)                       # kode data latih
        ref = Z[:2000]
        d_ref = torch.cdist(ref, Z)
        d_ref[torch.arange(2000), torch.arange(2000)] = float("inf")
        scale = d_ref.min(1).values.median().item()
        za, zb = m.encode(xa.to(device)).cpu(), m.encode(xb.to(device)).cpu()
        path = torch.cat([(1 - a) * za + a * zb for a in alphas])
        res[name] = (torch.cdist(path, Z).min(1).values / scale).tolist()
    return res


# ------------------------------------------------------------------ figur
def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def line_pair(ax, a, b, la, lb, title, ylabel):
    ep = range(1, len(a) + 1)
    ax.plot(ep, a, color=BLUE, lw=2, label=la)
    ax.plot(ep, b, color=ORANGE, lw=2, label=lb)
    # label langsung di ujung garis hanya jika kedua ujung cukup terpisah (hindari tabrakan)
    lo, hi = min(min(a), min(b)), max(max(a), max(b))
    if abs(a[-1] - b[-1]) > 0.08 * (hi - lo + 1e-12):
        for ys, lab in ((a, la), (b, lb)):
            ax.annotate(lab, (len(ys), ys[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=INK2, va="center")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel("epoch", fontsize=7, color=INK2)
    ax.set_ylabel(ylabel, fontsize=7, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    style_axes(ax)
    ax.set_xlim(1, len(a) * 1.18)


def fig1(models, x_sel, device):
    with torch.no_grad():
        rows = [("Asli", x_sel.to(device)),
                ("Vanilla AE", models["vanilla_ae"](x_sel.to(device))),
                ("Sparse AE", models["sparse_ae"](x_sel.to(device))),
                ("VAE (mean z)", models["vae"].reconstruct(x_sel.to(device)))]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 4.9))
    for ax, (lab, imgs) in zip(axes, rows):
        show_grid(ax, imgs, len(imgs))
        ax.set_title(lab, fontsize=9, loc="left", color=INK)
    fig.suptitle("Figur 1. Rekonstruksi citra uji (satu per kelas)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_reconstruction.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig2(samples, fids):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, name in zip(axes, GEN_MODELS):
        show_grid(ax, samples[name][:16], 4)
        sub = f"  (FID {fids[name]:.1f})" if fids and name in fids else ""
        ax.set_title(LABEL[name] + sub, fontsize=10, color=INK)
    fig.suptitle("Figur 2. Sampel acak dari noise / prior (4×4)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_samples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def fig4(models, xa, xb, alphas, device, ca, cb):
    rows = []
    for name in AE_MODELS:
        m = models[name]
        za, zb = m.encode(xa.to(device)), m.encode(xb.to(device))
        rows.append((LABEL[name] + f"  (encode: {ca} → {cb})",
                     torch.cat([m.decode((1 - a) * za + a * zb) for a in alphas])))
    g = torch.Generator().manual_seed(SEED)
    za, zb = torch.randn(1, Z_DIM_GAN, generator=g), torch.randn(1, Z_DIM_GAN, generator=g)
    for name in ("vanilla_gan", "dcgan"):
        G = models[name]
        rows.append((LABEL[name] + "  (z_A, z_B ~ N(0, I))",
                     torch.cat([to_zero1(G(((1 - a) * za + a * zb).to(device))) for a in alphas])))
    fig, axes = plt.subplots(len(rows), 1, figsize=(6.2, 8.6))
    for ax, (lab, imgs) in zip(axes, rows):
        show_grid(ax, imgs, len(alphas))
        ax.set_title(lab, fontsize=8.5, loc="left", color=INK)
    fig.suptitle("Figur 4. Interpolasi laten z(α) = (1−α)·z_A + α·z_B,  α = "
                 + ", ".join(f"{a:.1f}" for a in alphas), fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_interpolation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig4b(gap, alphas):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    for ax, name in zip(axes, AE_MODELS):
        ax.plot(alphas, gap[name], color=BLUE, lw=2, marker="o", ms=5)
        ax.axhline(1.0, color=INK2, lw=1, ls="--")
        ax.annotate("jarak NN tipikal", (0.5, 1.0), xytext=(0, -11), textcoords="offset points",
                    fontsize=7, color=INK2, ha="center")
        ax.set_ylim(bottom=0)
        ax.set_title(LABEL[name], fontsize=9, loc="left", color=INK)
        ax.set_xlabel("α", fontsize=8, color=INK2)
        style_axes(ax)
    axes[0].set_ylabel("jarak z(α) ke kode latih terdekat\n(÷ median jarak NN)", fontsize=7.5, color=INK2)
    fig.suptitle("Figur 4b. Seberapa jauh lintasan interpolasi dari data (nilai tinggi = wilayah kosong)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4b_latent_distance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig5(hist):
    fig = plt.figure(figsize=(13, 13.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 2.1], hspace=0.45, wspace=0.35)
    h = hist
    line_pair(fig.add_subplot(gs[0, 0]), h["vanilla_ae"]["train_loss"], h["vanilla_ae"]["val_loss"],
              "train", "val", "Vanilla AE", "MSE")
    line_pair(fig.add_subplot(gs[0, 1]), h["sparse_ae"]["train_loss"], h["sparse_ae"]["val_loss"],
              "train", "val", "Sparse AE", "MSE + λ‖h‖₁")
    line_pair(fig.add_subplot(gs[0, 2]), h["vae"]["train_loss"], h["vae"]["val_loss"],
              "train", "val", "VAE", "BCE + KL (per sampel)")
    line_pair(fig.add_subplot(gs[0, 3]), h["ddpm"]["train_loss"], h["ddpm"]["val_loss"],
              "train", "val", "DDPM", "‖ε − ε_θ‖²")
    for col, name in ((0, "vanilla_gan"), (1, "dcgan")):
        line_pair(fig.add_subplot(gs[1, col]), h[name]["d_loss"], h[name]["g_loss"],
                  "D", "G", f"{LABEL[name]}: loss", "BCE")
        ax = fig.add_subplot(gs[1, col + 2])
        line_pair(ax, h[name]["d_real"], h[name]["d_fake"], "D(x)", "D(G(z))",
                  f"{LABEL[name]}: output D", "probabilitas")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax = fig.add_subplot(gs[2, :])
    traj = FIG_DIR / "ddpm_trajectory.png"
    if traj.exists():
        ax.imshow(plt.imread(traj))
    ax.axis("off")
    ax.set_title("Trajektori denoising DDPM (x_T → x_0)", fontsize=9, loc="left", color=INK)
    fig.suptitle("Figur 5. Kurva training keenam model generatif dan trajektori denoising DDPM", fontsize=12)
    fig.savefig(FIG_DIR / "fig5_training.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figx_sparse(models, stats):
    W_ae = models["vanilla_ae"].encoder.net[1].weight.detach().cpu()
    W_sp = models["sparse_ae"].enc_sparse[1].weight.detach().cpu()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, W, name in ((axes[0], W_ae, "vanilla_ae"), (axes[1], W_sp, "sparse_ae")):
        w = W[:32].view(32, 1, 28, 28)
        lo = w.flatten(1).min(1).values.view(-1, 1, 1, 1)
        hi = w.flatten(1).max(1).values.view(-1, 1, 1, 1)
        show_grid(ax, (w - lo) / (hi - lo).clamp_min(1e-8), 8)
        s = stats[name]
        ax.set_title(f"{LABEL[name]}: 32 filter pertama lapisan 256\n"
                     f"rata-rata unit aktif/citra {s['mean_active_units_per_image']:.0f}/256, "
                     f"|h| rata-rata {s['mean_abs_activation']:.3f}", fontsize=8.5, color=INK)
    fig.suptitle("Fitur encoder (bobot lapisan pertama): Vanilla AE vs Sparse AE", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figx_sparse_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ tabel
def build_table(R, fids, models, mlp):
    rows = []
    for name in ("vanilla_ae", "sparse_ae", "vae", "vanilla_gan", "dcgan", "ddpm"):
        r = R[name]
        lat = r.get("latency_recon_ms_per_batch64") if name in ("vanilla_ae", "sparse_ae") else r["latency_ms_per_batch64"]
        if name in ("vanilla_ae", "sparse_ae", "vae"):
            note = "konvergen mulus (train≈val)"
        elif name == "ddpm":
            cov = r.get("mode_coverage") or {}
            note = (f"loss turun mulus; coverage {cov.get('classes_over_2pct', '-')}/10, "
                    f"entropi {cov.get('normalized_entropy', float('nan')):.3f}")
        else:
            hh, cov = r["history"], r.get("mode_coverage") or {}
            note = (f"D(x)={hh['d_real'][-1]:.2f}, D(G(z))={hh['d_fake'][-1]:.2f} di akhir; "
                    f"coverage {cov.get('classes_over_2pct', '-')}/10, "
                    f"entropi {cov.get('normalized_entropy', float('nan')):.3f}")
        rows.append({
            "Model": LABEL[name],
            "Test MSE": r.get("test_mse"), "PSNR (dB)": r.get("test_psnr"), "SSIM": r.get("test_ssim"),
            "FID": fids.get(name) if fids else None,
            "Waktu/epoch (s)": r["avg_epoch_time_s"],
            "Latensi 64 sampel (ms)": lat,
            "Parameter": n_params(models[name]),
            "Catatan kestabilan": note,
        })
    fd = mlp["frozen_decoder_test"]
    rows.append({"Model": LABEL["mlp_frozen_decoder"], "Test MSE": fd["mse"], "PSNR (dB)": fd["psnr"],
                 "SSIM": fd["ssim"], "FID": None, "Waktu/epoch (s)": None, "Latensi 64 sampel (ms)": None,
                 "Parameter": None,
                 "Catatan kestabilan": f"classifier test acc {mlp['mlp_test_accuracy'] * 100:.2f}%"})
    if fids and "real_train" in fids:
        rows.append({"Model": LABEL["real_train"], "Test MSE": None, "PSNR (dB)": None, "SSIM": None,
                     "FID": fids["real_train"], "Waktu/epoch (s)": None, "Latensi 64 sampel (ms)": None,
                     "Parameter": None, "Catatan kestabilan": "batas bawah FID (acuan)"})

    def fmt(k, v):
        if v is None:
            return "–"
        if k == "Parameter":
            return f"{v / 1e6:.2f} M"
        if k == "Test MSE":
            return f"{v:.5f}"
        if k == "SSIM":
            return f"{v:.4f}"
        if k == "Latensi 64 sampel (ms)":
            return f"{v:,.2f}" if v < 100 else f"{v:,.0f}"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    keys = list(rows[0].keys())
    md = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    md += ["| " + " | ".join(fmt(k, r[k]) for k in keys) + " |" for r in rows]
    (RESULTS_DIR / "metrics_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(RESULTS_DIR / "metrics_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-fid", type=int, default=5000)
    p.add_argument("--ddpm-bs", type=int, default=250, help="batch sampling DDPM (kecilkan jika OOM)")
    p.add_argument("--skip-fid", action="store_true")
    args = p.parse_args()

    device = get_device()
    print("Device:", device)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models(device)

    train_loader, _, test_loader = get_loaders()
    x_test = torch.cat([x for x, _ in test_loader])
    y_test = torch.cat([y for _, y in test_loader])
    x_train = torch.cat([x for x, _ in train_loader])[:10000]   # shuffle=True -> subset acak

    R = {n: json.loads((RESULTS_DIR / f"{n}.json").read_text())
         for n in ("vanilla_ae", "sparse_ae", "vae", "vanilla_gan", "dcgan", "ddpm")}
    mlp = json.loads((RESULTS_DIR / "mlp_experiment.json").read_text())
    prev = json.loads((RESULTS_DIR / "eval.json").read_text()) if (RESULTS_DIR / "eval.json").exists() else {}

    # --- sampel + FID
    samples = {n: generate(n, models[n], args.n_fid, device, args.ddpm_bs) for n in GEN_MODELS}
    if args.skip_fid:
        fids = prev.get("fid")
    else:
        fids = compute_fid(samples, x_test, x_train[:args.n_fid], device)

    # --- analisis
    idx10 = torch.stack([(y_test == c).nonzero()[0, 0] for c in range(10)])
    alphas = [round(0.2 * i, 1) for i in range(6)]
    ca, cb = 1, 9   # Trouser -> Ankle boot (bentuk sangat berbeda)
    ia, ib = (y_test == ca).nonzero()[0, 0], (y_test == cb).nonzero()[0, 0]
    xa, xb = x_test[ia:ia + 1], x_test[ib:ib + 1]
    sp = sparsity_stats(models, x_test, device)
    gap = latent_gap(models, x_train, xa, xb, device, alphas)

    # --- figur
    fig1(models, x_test[idx10], device)
    fig2(samples, fids)
    fig4(models, xa, xb, alphas, device, CLASS_NAMES[ca], CLASS_NAMES[cb])
    fig4b(gap, alphas)
    hist = {n: R[n]["history"] for n in R}
    fig5(hist)
    figx_sparse(models, sp)

    build_table(R, fids, models, mlp)
    save_json({"fid": fids,
               "fid_protocol": {"n_generated": args.n_fid, "real_reference": "test set (7000)",
                                "feature": "InceptionV3 pool3 2048", "preprocess": "gray->3ch, resize 299"},
               "sparsity": sp,
               "interpolation": {"alphas": alphas, "pair": [CLASS_NAMES[ca], CLASS_NAMES[cb]],
                                 "latent_gap_normalized": gap},
               "params": {n: n_params(models[n]) for n in models}},
              RESULTS_DIR / "eval.json")
    print("Selesai. Lihat results/metrics_table.md dan results/figures/")


if __name__ == "__main__":
    main()
