"""Diagram arsitektur ketujuh model (untuk Bagian 2 laporan).

    python make_arch_diagrams.py
Output: results/figures/arch_1_autoencoders.png, arch_2_gans.png, arch_3_ddpm.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "figures"
INK, INK2 = "#0b0b0b", "#52514e"
FILL = {"io": "#eeede9", "enc": "#dbe8f8", "lat": "#fde3d8", "dec": "#dff0e4", "head": "#ece3f6"}
EDGE = {"io": "#8a8985", "enc": "#2a78d6", "lat": "#eb6834", "dec": "#2f8f55", "head": "#7d4fb3"}


def box(ax, x, y, w, h, text, kind):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.6, color=INK, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, style="-|>", color=INK2, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.2, ls=ls, shrinkA=0, shrinkB=0))


def chain(ax, y, items, x0=1.2, x1=13.8, w=1.35, h=0.62):
    """items: list of (text, kind). Mengembalikan posisi x tiap kotak."""
    n = len(items)
    xs = [x0 + i * (x1 - x0) / (n - 1) for i in range(n)] if n > 1 else [(x0 + x1) / 2]
    for i, (x, (t, k)) in enumerate(zip(xs, items)):
        box(ax, x, y, w, h, t, k)
        if i:
            arrow(ax, xs[i - 1] + w / 2, y, x - w / 2, y)
    return xs


def title(ax, y, name, loss):
    ax.text(0.05, y + 0.62, name, fontsize=10, fontweight="bold", color=INK, va="bottom")
    ax.text(14.95, y + 0.62, "Loss: " + loss, fontsize=8.5, color=INK2, va="bottom", ha="right")


def new_fig(h):
    fig, ax = plt.subplots(figsize=(13, h))
    ax.set_xlim(0, 15)
    ax.axis("off")
    return fig, ax


def legend(ax, y, kinds):
    names = {"io": "input / output", "enc": "encoder / generator", "lat": "ruang laten",
             "dec": "decoder", "head": "diskriminator / classifier"}
    x = 0.3
    for k in kinds:
        ax.add_patch(FancyBboxPatch((x, y - 0.12), 0.35, 0.24, boxstyle="round,pad=0.01",
                                    fc=FILL[k], ec=EDGE[k], lw=1.2))
        ax.text(x + 0.45, y, names[k], fontsize=7.5, va="center", color=INK2)
        x += 0.45 + 0.085 * len(names[k]) + 0.5


def save(fig, ax, ymin, ymax, name):
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_autoencoders():
    fig, ax = new_fig(9.2)
    y = 8.6
    title(ax, y, "(a) Vanilla Autoencoder", "MSE(x, x̂)")
    chain(ax, y, [("x\n784", "io"), ("Linear\n784→256\nReLU", "enc"), ("Linear\n256→64", "enc"),
                  ("z\n64", "lat"), ("Linear\n64→256\nReLU", "dec"), ("Linear\n256→784\nSigmoid", "dec"),
                  ("x̂\n784", "io")])

    y = 6.4
    title(ax, y, "(b) Sparse Autoencoder", "MSE(x, x̂) + λ·mean|h|,  λ = 10⁻³")
    xs = chain(ax, y, [("x\n784", "io"), ("Linear\n784→256\nReLU = h", "enc"), ("Linear\n256→64", "enc"),
                       ("z\n64", "lat"), ("Linear\n64→256\nReLU", "dec"), ("Linear\n256→784\nSigmoid", "dec"),
                       ("x̂\n784", "io")])
    ax.text(xs[1], y - 0.48, "penalti L1 pada h", ha="center", va="top", fontsize=7.3, color=EDGE["lat"])

    y = 4.2
    title(ax, y, "(c) Variational Autoencoder", "BCE(x, x̂) + β·KL(q(z|x) ‖ N(0, I)),  β = 1")
    xs = [1.2, 3.3, 5.4, 7.5, 9.6, 11.7, 13.8]
    box(ax, xs[0], y, 1.35, 0.62, "x\n784", "io")
    box(ax, xs[1], y, 1.35, 0.62, "Linear\n784→256\nReLU", "enc")
    box(ax, xs[2], y + 0.42, 1.35, 0.42, "μ  (Linear 256→64)", "lat")
    box(ax, xs[2], y - 0.42, 1.35, 0.42, "log σ²  (256→64)", "lat")
    box(ax, xs[3], y, 1.35, 0.62, "z = μ + σ⊙ε\nε ~ N(0, I)", "lat")
    box(ax, xs[4], y, 1.35, 0.62, "Linear\n64→256\nReLU", "dec")
    box(ax, xs[5], y, 1.35, 0.62, "Linear\n256→784\nSigmoid", "dec")
    box(ax, xs[6], y, 1.35, 0.62, "x̂\n784", "io")
    arrow(ax, xs[0] + 0.675, y, xs[1] - 0.675, y)
    for dy in (0.42, -0.42):
        arrow(ax, xs[1] + 0.675, y, xs[2] - 0.675, y + dy)
        arrow(ax, xs[2] + 0.675, y + dy, xs[3] - 0.675, y)
    for a, b in ((3, 4), (4, 5), (5, 6)):
        arrow(ax, xs[a] + 0.675, y, xs[b] - 0.675, y)

    y = 2.0
    title(ax, y, "(d) MLP Classifier (Bagian B)", "Cross-Entropy(y, ŷ)")
    xs = chain(ax, y, [("x\n784", "io"), ("Linear\n784→256\nReLU", "enc"), ("Linear\n256→64\nReLU", "enc"),
                       ("z_MLP\n64", "lat"), ("Linear\n64→10", "head"), ("softmax\nP(y|x)", "head"),
                       ("ŷ\n10 kelas", "io")])
    ax.text((xs[1] + xs[2]) / 2, y - 0.48, "struktur identik dengan encoder Vanilla AE",
            ha="center", va="top", fontsize=7.3, color=INK2)
    legend(ax, 0.7, ["io", "enc", "lat", "dec", "head"])
    save(fig, ax, 0.4, 9.6, "arch_1_autoencoders.png")


def fig_gans():
    fig, ax = new_fig(7.6)
    y = 7.0
    title(ax, y, "(e) Vanilla GAN — Generator", "BCE non-saturating: −log D(G(z))")
    chain(ax, y, [("z ~ N(0, I)\n100", "lat"), ("Linear 100→256\nLeakyReLU", "enc"),
                  ("Linear 256→512\nLeakyReLU", "enc"), ("Linear 512→1024\nLeakyReLU", "enc"),
                  ("Linear 1024→784\nTanh", "enc"), ("G(z)\n1×28×28", "io")], w=1.75)
    y = 5.4
    title(ax, y, "(e) Vanilla GAN — Diskriminator", "BCE: −[log D(x) + log(1 − D(G(z)))]")
    chain(ax, y, [("x atau G(z)\n784", "io"), ("Linear 784→512\nLeakyReLU\nDropout 0,3", "head"),
                  ("Linear 512→256\nLeakyReLU\nDropout 0,3", "head"), ("Linear 256→1", "head"),
                  ("logit\nasli / palsu", "io")], w=1.85, h=0.72)
    y = 3.4
    title(ax, y, "(f) DCGAN — Generator", "BCE non-saturating: −log D(G(z))")
    chain(ax, y, [("z ~ N(0, I)\n100", "lat"), ("Linear 100→6272\nBN, ReLU\nreshape 128×7×7", "enc"),
                  ("ConvT 4×4, s2\n128→64\n64×14×14, BN, ReLU", "enc"),
                  ("ConvT 4×4, s2\n64→1\n1×28×28, Tanh", "enc"), ("G(z)\n1×28×28", "io")], w=2.0, h=0.78)
    y = 1.6
    title(ax, y, "(f) DCGAN — Diskriminator", "BCE: −[log D(x) + log(1 − D(G(z)))]")
    chain(ax, y, [("x atau G(z)\n1×28×28", "io"), ("Conv 4×4, s2\n1→64\n64×14×14, LeakyReLU", "head"),
                  ("Conv 4×4, s2\n64→128\n128×7×7, BN, LeakyReLU", "head"), ("Flatten\nLinear 6272→1", "head"),
                  ("logit\nasli / palsu", "io")], w=2.1, h=0.78)
    legend(ax, 0.55, ["io", "lat", "enc", "head"])
    save(fig, ax, 0.25, 7.9, "arch_2_gans.png")


def fig_ddpm():
    fig, ax = new_fig(6.4)
    yt, yb = 4.6, 1.9                       # baris encoder (atas) dan decoder (bawah)
    title(ax, 5.25, "(g) DDPM — U-Net ε_θ(x_t, t)   (3,30 juta parameter)", "‖ε − ε_θ(x_t, t)‖²,  T = 1000")
    W, H = 1.6, 0.8
    enc = [(1.2, "x_t\n1×28×28", "io"), (3.4, "Conv 3×3\n1→64\n64×28×28", "enc"),
           (5.6, "ResBlock\n64×28×28", "enc"), (7.8, "↓ Conv s2\nResBlock\n128×14×14", "enc"),
           (10.0, "↓ Conv s2\nResBlock + Attn\n128×7×7", "enc")]
    for i, (x, t, k) in enumerate(enc):
        box(ax, x, yt, W, H, t, k)
        if i:
            arrow(ax, enc[i - 1][0] + W / 2, yt, x - W / 2, yt)
    box(ax, 12.5, (yt + yb) / 2, 1.9, 1.15, "Bottleneck\nResBlock → Attn\n→ ResBlock\n128×7×7", "lat")
    arrow(ax, 10.0 + W / 2, yt, 12.5 - 0.95, (yt + yb) / 2 + 0.3)
    dec = [(10.0, "concat skip\nResBlock + Attn\n128×7×7", "dec"), (7.8, "↑ Upsample\nconcat skip\nResBlock 128×14×14", "dec"),
           (5.6, "↑ Upsample\nconcat skip\nResBlock 64×28×28", "dec"), (3.4, "GroupNorm, SiLU\nConv 3×3\n64→1", "dec"),
           (1.2, "ε̂\n1×28×28", "io")]
    arrow(ax, 12.5 - 0.95, (yt + yb) / 2 - 0.3, 10.0 + W / 2, yb)
    for i, (x, t, k) in enumerate(dec):
        box(ax, x, yb, W, H + 0.1, t, k)
        if i:
            arrow(ax, dec[i - 1][0] - W / 2, yb, x + W / 2, yb)
    for x in (10.0, 7.8, 5.6):               # skip connection
        arrow(ax, x, yt - H / 2, x, yb + (H + 0.1) / 2, color=EDGE["lat"], ls="--")
    ax.text(8.95, (yt + yb) / 2, "skip connection", fontsize=7.3, color=EDGE["lat"], va="center", ha="center")
    box(ax, 2.3, 3.35, 2.6, 0.5, "t → embedding sinusoidal → MLP", "lat")
    ax.text(2.3, 2.95, "disuntikkan ke setiap ResBlock", fontsize=7.3, color=INK2, va="center", ha="center")
    ax.text(0.3, 0.75, "Forward: x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε,  β linear 10⁻⁴…0,02.   "
            "Sampling: x_T ~ N(0, I) → 1000 langkah p_θ(x_{t−1}|x_t) → x_0.   Bobot EMA 0,999 dipakai untuk sampling.",
            fontsize=8, color=INK2)
    legend(ax, 0.25, ["io", "enc", "lat", "dec"])
    save(fig, ax, -0.05, 5.8, "arch_3_ddpm.png")


if __name__ == "__main__":
    fig_autoencoders()
    fig_gans()
    fig_ddpm()
    print("Diagram tersimpan di", OUT)
