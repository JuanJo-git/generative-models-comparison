| Model | Test MSE | PSNR (dB) | SSIM | FID | Waktu/epoch (s) | Latensi 64 sampel (ms) | Parameter | Catatan kestabilan |
|---|---|---|---|---|---|---|---|---|
| Vanilla AE | 0.00649 | 23.02 | 0.8299 | – | 8.82 | 0.80 | 0.44 M | konvergen mulus (train≈val) |
| Sparse AE | 0.00678 | 22.78 | 0.8251 | – | 10.15 | 0.80 | 0.44 M | konvergen mulus (train≈val) |
| VAE | 0.01335 | 19.59 | 0.7060 | 87.88 | 10.50 | 0.40 | 0.45 M | konvergen mulus (train≈val) |
| Vanilla GAN | – | – | – | 143.88 | 4.86 | 0.37 | 1.49 M | D(x)=0.53, D(G(z))=0.47 di akhir; coverage 10/10, entropi 0.996 |
| DCGAN | – | – | – | 24.11 | 8.63 | 1.01 | 0.77 M | D(x)=0.69, D(G(z))=0.31 di akhir; coverage 10/10, entropi 0.994 |
| DDPM | – | – | – | 11.28 | 79.59 | 31,475 | 3.30 M | loss turun mulus; coverage 10/10, entropi 0.994 |
| MLP + frozen decoder | 0.01732 | 18.39 | 0.6578 | – | – | – | – | classifier test acc 88.77% |
| Data asli (train) | – | – | – | 4.32 | – | – | – | batas bawah FID (acuan) |
