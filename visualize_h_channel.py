"""Visualize and compare the Hematoxylin (H) channel of paired H&E and IHC
images from the train dataset using per-image Macenko stain separation.

Output figures
--------------
1. h_channel_comparison.png   – per-sample grid (5 cols): HE RGB | HE H-map |
                                 IHC RGB | IHC H-map | H diff (IHC − HE)
2. h_rendered_comparison.png  – same grid but H-channel rendered back to RGB
                                 (shows the synthetic hematoxylin-stained image)
3. h_channel_distribution.png – overlaid H-amount histograms (H&E vs IHC)
4. stain_vectors.png          – estimated H and DAB OD unit vectors per image
5. h_scatter.png              – per-pixel H-amount scatter (H&E vs IHC, sampled)

Usage
-----
    python visualize_h_channel.py [--n_samples N] [--output_dir PATH]
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from skimage.exposure import rescale_intensity

# ── add project root so src.stains.macenko is importable ──────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.stains.macenko import estimate_stain_matrix, rgb_to_od  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
HE_DIR  = ROOT / "data/BCI/train/HE"
IHC_DIR = ROOT / "data/BCI/train/IHC"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_rgb_tensor(path: Path) -> torch.Tensor:
    """Load PNG as float32 (3,H,W) in [0,1]."""
    img = io.imread(str(path)).astype(np.float32) / 255.0
    if img.ndim == 2:                          # grayscale → RGB
        img = np.stack([img] * 3, axis=2)
    if img.shape[2] == 4:                      # drop alpha
        img = img[..., :3]
    return torch.from_numpy(img.transpose(2, 0, 1))  # (3,H,W)


def extract_h_map(img_01: torch.Tensor) -> torch.Tensor:
    """
    Return the per-pixel Hematoxylin amount (H,W) using per-image Macenko.
    Uses pinv(V) @ OD, then takes stain index 0 (H column).
    """
    batch = img_01.unsqueeze(0)                        # (1,3,H,W)
    V_list = estimate_stain_matrix(batch, reference_mode="none")
    V = V_list[0]                                      # (3,2) – cols: [H, DAB]

    od = rgb_to_od(img_01)                             # (3,H,W)
    H_px, W_px = od.shape[1], od.shape[2]
    od_flat = od.reshape(3, -1)                        # (3,HW)

    V_pinv = torch.linalg.pinv(V)                     # (2,3)
    S = V_pinv @ od_flat                               # (2,HW)

    h_map = S[0].reshape(H_px, W_px).clamp(min=0)     # (H,W) non-negative H
    return h_map, V


def render_h_rgb(h_map: torch.Tensor, V: torch.Tensor) -> np.ndarray:
    """Reconstruct the synthetic hematoxylin-only RGB image.

    od_H = h_map * V[:,0]  →  I = 10^(-od_H), clipped to [0,1].
    Returns (H,W,3) float32.
    """
    h_vec = V[:, 0]                                    # (3,) H stain OD vector
    H_px, W_px = h_map.shape
    od_h = h_map.reshape(-1).unsqueeze(0) * h_vec.unsqueeze(1)  # (3, HW)
    od_h = od_h.reshape(3, H_px, W_px)
    rgb = torch.pow(10.0, -od_h).clamp(0.0, 1.0)
    return rgb.permute(1, 2, 0).numpy()


def to_display(arr: np.ndarray) -> np.ndarray:
    """Rescale a float array to [0,1] for grayscale display."""
    return rescale_intensity(arr, out_range=(0.0, 1.0))


def paired_stems(n: int, seed: int = 42) -> list:
    he_stems  = {p.stem for p in HE_DIR.glob("*.png")}
    ihc_stems = {p.stem for p in IHC_DIR.glob("*.png")}
    common    = sorted(he_stems & ihc_stems)
    if not common:
        raise FileNotFoundError(
            f"No shared filenames between {HE_DIR} and {IHC_DIR}."
        )
    rng = random.Random(seed)
    rng.shuffle(common)
    return common[:n]


# ── figure 1: H-amount map comparison ────────────────────────────────────────

def fig_h_channel_comparison(stems, out_path):
    """5 columns: HE RGB | HE H-map | IHC RGB | IHC H-map | H diff (IHC−HE)."""
    n = len(stems)
    fig, axes = plt.subplots(n, 5, figsize=(20, 3.8 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = [
        "H&E (RGB)", "H&E – H amount", "IHC (RGB)", "IHC – H amount",
        "H diff (IHC − H&E)",
    ]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10, fontweight="bold")

    for row, stem in enumerate(stems):
        he_t  = load_rgb_tensor(HE_DIR  / f"{stem}.png")
        ihc_t = load_rgb_tensor(IHC_DIR / f"{stem}.png")

        he_h,  _  = extract_h_map(he_t)
        ihc_h, _  = extract_h_map(ihc_t)

        he_h_np  = he_h.numpy()
        ihc_h_np = ihc_h.numpy()
        diff      = ihc_h_np - he_h_np
        abs_max   = max(float(np.abs(diff).max()), 1e-6)

        panels = [
            (he_t.permute(1,2,0).numpy(),  "gray", None,              False),
            (to_display(he_h_np),          "gray", (0, 1),            False),
            (ihc_t.permute(1,2,0).numpy(), "gray", None,              False),
            (to_display(ihc_h_np),         "gray", (0, 1),            False),
            (diff,                         "RdBu_r", (-abs_max, abs_max), True),
        ]

        for col, (img, cmap, clim, add_cbar) in enumerate(panels):
            ax = axes[row, col]
            kw = {"cmap": cmap}
            if clim is not None:
                kw["vmin"], kw["vmax"] = clim
            im = ax.imshow(img, **kw)
            ax.axis("off")
            if add_cbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

        axes[row, 0].set_ylabel(stem[:20], fontsize=7, rotation=0, labelpad=65, va="center")

    fig.suptitle("Macenko H-channel: H&E vs IHC (train set)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── figure 2: rendered H-stain RGB comparison ─────────────────────────────────

def fig_h_rendered_comparison(stems, out_path):
    """5 columns: HE RGB | HE H-render | IHC RGB | IHC H-render | render diff."""
    n = len(stems)
    fig, axes = plt.subplots(n, 5, figsize=(20, 3.8 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = [
        "H&E (RGB)", "H&E – H rendered", "IHC (RGB)", "IHC – H rendered",
        "Rendered diff (IHC − H&E)",
    ]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10, fontweight="bold")

    for row, stem in enumerate(stems):
        he_t  = load_rgb_tensor(HE_DIR  / f"{stem}.png")
        ihc_t = load_rgb_tensor(IHC_DIR / f"{stem}.png")

        he_h,  he_V  = extract_h_map(he_t)
        ihc_h, ihc_V = extract_h_map(ihc_t)

        he_rend  = render_h_rgb(he_h,  he_V)
        ihc_rend = render_h_rgb(ihc_h, ihc_V)

        # Luminance diff of rendered images
        lum_he  = he_rend.mean(axis=2)
        lum_ihc = ihc_rend.mean(axis=2)
        diff     = lum_ihc - lum_he
        abs_max  = max(float(np.abs(diff).max()), 1e-6)

        panels = [
            (he_t.permute(1,2,0).numpy(),  None,      None,               False),
            (he_rend,                      None,      None,               False),
            (ihc_t.permute(1,2,0).numpy(), None,      None,               False),
            (ihc_rend,                     None,      None,               False),
            (diff,                         "RdBu_r",  (-abs_max, abs_max), True),
        ]

        for col, (img, cmap, clim, add_cbar) in enumerate(panels):
            ax = axes[row, col]
            kw = {} if cmap is None else {"cmap": cmap}
            if clim is not None:
                kw["vmin"], kw["vmax"] = clim
            im = ax.imshow(img, **kw)
            ax.axis("off")
            if add_cbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

        axes[row, 0].set_ylabel(stem[:20], fontsize=7, rotation=0, labelpad=65, va="center")

    fig.suptitle("Macenko H rendered back to RGB: H&E vs IHC", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── figure 3: distribution histograms ────────────────────────────────────────

def fig_distribution(stems, out_path):
    he_vals, ihc_vals = [], []
    for stem in stems:
        he_h,  _ = extract_h_map(load_rgb_tensor(HE_DIR  / f"{stem}.png"))
        ihc_h, _ = extract_h_map(load_rgb_tensor(IHC_DIR / f"{stem}.png"))
        he_vals.append(he_h.numpy().ravel())
        ihc_vals.append(ihc_h.numpy().ravel())

    he_all  = np.concatenate(he_vals)
    ihc_all = np.concatenate(ihc_vals)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Overlaid histogram
    lo = min(he_all.min(), ihc_all.min())
    hi = max(np.percentile(he_all, 99.5), np.percentile(ihc_all, 99.5))
    bins = np.linspace(lo, hi, 120)
    axes[0].hist(he_all,  bins=bins, alpha=0.6, color="steelblue", label="H&E",  density=True)
    axes[0].hist(ihc_all, bins=bins, alpha=0.6, color="coral",     label="IHC",  density=True)
    for arr, color, label in [(he_all, "steelblue", "H&E"), (ihc_all, "coral", "IHC")]:
        med = float(np.median(arr))
        axes[0].axvline(med, color=color, linestyle="--", linewidth=1.5,
                        label=f"{label} median={med:.3f}")
    axes[0].set_xlabel("H-amount (OD units, clipped ≥ 0)", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].set_title(f"H-channel distribution (n={len(stems)} samples)", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Box plot
    axes[1].boxplot(
        [he_all[::50], ihc_all[::50]],   # subsample to keep it fast
        labels=["H&E", "IHC"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[1].set_ylabel("H-amount (OD units)", fontsize=10)
    axes[1].set_title("H-channel boxplot (subsampled)", fontsize=11)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Macenko H-channel statistics: H&E vs IHC", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── figure 4: stain vectors ───────────────────────────────────────────────────

def fig_stain_vectors(stems, out_path):
    """Visualise estimated H and DAB OD unit vectors for each sample."""
    he_H, he_D, ihc_H, ihc_D = [], [], [], []
    labels_he, labels_ihc = [], []

    for stem in stems:
        _, he_V  = extract_h_map(load_rgb_tensor(HE_DIR  / f"{stem}.png"))
        _, ihc_V = extract_h_map(load_rgb_tensor(IHC_DIR / f"{stem}.png"))
        he_H.append(he_V[:, 0].numpy())
        he_D.append(he_V[:, 1].numpy())
        ihc_H.append(ihc_V[:, 0].numpy())
        ihc_D.append(ihc_V[:, 1].numpy())
        labels_he.append(f"HE {stem[:12]}")
        labels_ihc.append(f"IHC {stem[:12]}")

    channels = ["R", "G", "B"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    titles = [("H&E – H stain vector", he_H, "steelblue"),
              ("H&E – DAB stain vector", he_D, "goldenrod"),
              ("IHC – H stain vector", ihc_H, "coral"),
              ("IHC – DAB stain vector", ihc_D, "seagreen")]

    for ax, (title, vecs, color) in zip(axes.ravel(), titles):
        mat = np.stack(vecs)           # (N, 3)
        x   = np.arange(mat.shape[0])
        for ch, ch_name, ls in zip(range(3), channels, ["-", "--", ":"]):
            ax.plot(x, mat[:, ch], marker="o", markersize=4,
                    linestyle=ls, linewidth=1.5, label=ch_name, color=color,
                    alpha=[1.0, 0.7, 0.5][ch])
        mean_v = mat.mean(axis=0)
        ax.set_title(f"{title}\nmean=[{mean_v[0]:.3f}, {mean_v[1]:.3f}, {mean_v[2]:.3f}]",
                     fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([s[:10] for s in labels_he], rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("OD unit vector component", fontsize=9)
        ax.legend(title="Channel", fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.8)

    fig.suptitle("Macenko estimated stain vectors per sample", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── figure 5: per-pixel H scatter ─────────────────────────────────────────────

def fig_h_scatter(stems, out_path, max_pts=30_000):
    """Scatter plot: H-amount in H&E (x) vs H-amount in IHC (y), per pixel."""
    n = len(stems)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).ravel()

    for idx, stem in enumerate(stems):
        he_h,  _ = extract_h_map(load_rgb_tensor(HE_DIR  / f"{stem}.png"))
        ihc_h, _ = extract_h_map(load_rgb_tensor(IHC_DIR / f"{stem}.png"))

        he_np  = he_h.numpy().ravel()
        ihc_np = ihc_h.numpy().ravel()

        # Subsample for readability
        rng = np.random.default_rng(42)
        idx_s = rng.choice(len(he_np), size=min(max_pts, len(he_np)), replace=False)
        x, y = he_np[idx_s], ihc_np[idx_s]

        ax = axes[idx]
        ax.scatter(x, y, s=2, alpha=0.15, color="steelblue", rasterized=True)
        lim = float(max(np.percentile(np.concatenate([x, y]), 99), 1e-4))
        ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Equal")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

        r = float(np.corrcoef(x, y)[0, 1])
        ax.set_title(f"{stem[:18]}\nPearson r={r:.3f}", fontsize=9)
        ax.set_xlabel("H-amount H&E", fontsize=8)
        ax.set_ylabel("H-amount IHC", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    # Hide unused axes
    for idx in range(len(stems), len(axes)):
        axes[idx].set_axis_off()

    fig.suptitle("Per-pixel H-amount: H&E vs IHC (Macenko)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare Macenko H channel: H&E vs IHC")
    parser.add_argument("--n_samples", type=int, default=6)
    parser.add_argument("--output_dir", type=str, default="outputs/h_channel_viz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = paired_stems(args.n_samples, seed=args.seed)
    print(f"Selected {len(stems)} paired samples: {stems}")

    fig_h_channel_comparison(stems, out_dir / "h_channel_comparison.png")
    fig_h_rendered_comparison(stems, out_dir / "h_rendered_comparison.png")
    fig_distribution(stems, out_dir / "h_channel_distribution.png")
    fig_stain_vectors(stems, out_dir / "stain_vectors.png")
    fig_h_scatter(stems, out_dir / "h_scatter.png")

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
