# Studi Komparatif Model Generatif (Deep Learning D6421, Sesi 07)

Perbandingan empiris **Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, dan DDPM** pada
**Fashion-MNIST**, ditambah eksperimen pembuktian bahwa **MLP Classifier tidak dapat dipakai
untuk image generation**. Semua kode ditulis dengan PyTorch.

## Setup (Windows, VS Code)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
# PyTorch dengan CUDA (GPU NVIDIA); --no-cache-dir mencegah MemoryError pada RAM 8 GB
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # harus True
```

`requirements.txt` sengaja tidak memuat `torch`/`torchvision` agar instalasi PyTorch versi CUDA
tidak tertimpa versi CPU dari PyPI.

## Standardisasi eksperimen

| Aspek | Nilai |
|---|---|
| Dataset | Fashion-MNIST, gabungan 70.000 citra, split 80/10/10 (train/val/test) |
| Seed | 42 (`torch`, `numpy`, `random`); split memakai `torch.Generator` ber-seed |
| Batch size | 128 |
| Skala piksel | disimpan di `[0, 1]`; model ber-Tanh (GAN, DCGAN, DDPM) memakai `[-1, 1]` dan dikembalikan ke `[0, 1]` sebelum metrik |
| Dimensi laten | 64 (AE, Sparse AE, VAE, z_MLP); 100 untuk prior GAN/DCGAN |
| Hardware | NVIDIA GeForce GTX 1650 Ti (4 GB), Ryzen 5 4600H, RAM 8 GB |

## Urutan menjalankan

```powershell
python train_ae.py --model all --epochs 30     # Vanilla AE, Sparse AE, VAE
python experiment_mlp.py                       # Bagian B: MLP Classifier (5 tahap) + Figur 3
python train_gan.py --model all --epochs 30    # Vanilla GAN, DCGAN
python train_ddpm.py --epochs 30               # DDPM (tambahkan --resume jika terputus)
python evaluate.py                             # FID, tabel metrik, Figur 1, 2, 4, 5
```

`experiment_mlp.py` dijalankan sebelum `train_gan.py`/`train_ddpm.py` karena MLP Classifier
dipakai untuk mengukur *mode coverage* (sebaran kelas sampel generatif).

## Arsitektur ringkas

| Model | Arsitektur | Loss |
|---|---|---|
| Vanilla AE | 784→256→64→256→784 (ReLU, output Sigmoid) | MSE |
| Sparse AE | 784→256 (lapisan sparse, ReLU)→64→256→784 | MSE + λ·mean‖h‖₁, λ = 1e-3 |
| VAE | 784→256→(μ, log σ²)∈ℝ⁶⁴, reparameterization →256→784 | BCE + KL(q(z\|x) ‖ N(0, I)) |
| Vanilla GAN | G: 100→256→512→1024→784 (Tanh); D: 784→512→256→1 (LeakyReLU, Dropout) | BCE non-saturating |
| DCGAN | G: 100→128×7×7→ConvT→64×14×14→ConvT→1×28×28; D: Conv 28→14→7→Linear | BCE non-saturating |
| DDPM | U-Net 3,3 M param (28→14→7 + self-attention), T = 1000, β linear 1e-4..0.02, EMA 0.999 | ‖ε − ε_θ(x_t, t)‖² |
| MLP Classifier | 784→256→64→10 (ReLU) | Cross-Entropy |

Optimizer: Adam (AE/VAE/MLP lr 1e-3; GAN/DCGAN lr 2e-4, β₁ = 0,5; DDPM lr 2e-4).

## Output

| Lokasi | Isi |
|---|---|
| `results/*.json` | metrik test, riwayat loss per epoch, waktu/epoch, latensi, mode coverage |
| `results/eval.json` | FID, statistik sparsity, jarak laten interpolasi |
| `results/metrics_table.md` / `.csv` | tabel metrik lengkap |
| `results/figures/fig1_reconstruction.png` | Figur 1: rekonstruksi |
| `results/figures/fig2_samples.png` | Figur 2: sampel acak 4×4 |
| `results/figures/fig3_mlp_proof.png` | Figur 3: bukti MLP Classifier |
| `results/figures/fig4_interpolation.png`, `fig4b_latent_distance.png` | Figur 4: interpolasi laten |
| `results/figures/fig5_training.png` | Figur 5: kurva training + trajektori DDPM |
| `results/figures/gan_progress/` | sampel dari noise tetap tiap 5 epoch (GAN, DCGAN, DDPM) |

Checkpoint (`checkpoints/`) dan dataset (`data/`) tidak di-commit karena ukurannya.

## Struktur

```
src/
  config.py           seed, dataset, hyperparameter
  data.py             loader + split 80/10/10
  metrics.py          MSE, PSNR, SSIM
  utils.py            seed, device, checkpoint, pengukuran latensi
  viz.py              helper grid citra
  models/
    autoencoder.py    Vanilla AE, Sparse AE, VAE
    mlp.py            MLP Classifier
    gan.py            Vanilla GAN, DCGAN
    ddpm.py           U-Net, Gaussian diffusion, EMA
train_ae.py  experiment_mlp.py  train_gan.py  train_ddpm.py  evaluate.py
```
