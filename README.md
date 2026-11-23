# Studi Komparatif Model Generatif (Deep Learning D6421, Sesi 07)

Perbandingan empiris Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, dan DDPM pada
**Fashion-MNIST**, ditambah eksperimen pembuktian batasan MLP Classifier untuk image generation.

> Status: tahap 1 (fondasi + keluarga Autoencoder). Model GAN, DCGAN, DDPM, eksperimen MLP,
> FID, dan figur akan ditambahkan bertahap.

## Setup (Windows, VS Code)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
# PyTorch dengan CUDA (pilih perintah dari https://pytorch.org/get-started/locally/, mis. CUDA 12.4)
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # harus True
```

Catatan: `requirements.txt` tidak mengunci versi CUDA; install PyTorch lewat perintah di atas
terlebih dahulu agar versi CUDA yang terpasang.

## Standardisasi eksperimen

- Dataset: Fashion-MNIST (gabungan 70.000 citra, split 80/10/10 train/val/test)
- Seed tetap: `42` (`torch`, `numpy`, `random`)
- Batch size: 128 (shuffle hanya pada train)
- Piksel disimpan di `[0, 1]`; model ber-Tanh dikonversi ke `[-1, 1]` dan dikembalikan
  ke `[0, 1]` sebelum metrik dihitung
- Dimensi laten AE/Sparse AE/VAE = 64

## Menjalankan

```powershell
python train_ae.py --model all --epochs 30          # Vanilla AE, Sparse AE, VAE
python train_ae.py --model vae --epochs 30          # satu model saja
python train_ae.py --model all --epochs 1 --max-batches 20   # smoke test cepat
```

Output: checkpoint di `checkpoints/`, metrik + kurva loss (JSON) di `results/`.

## Struktur

```
src/
  config.py        seed, dataset, hyperparameter
  data.py          loader + split 80/10/10
  metrics.py       MSE, PSNR, SSIM
  utils.py         seed, device, checkpoint, ukur latensi
  models/
    autoencoder.py Vanilla AE, Sparse AE, VAE
train_ae.py
```
