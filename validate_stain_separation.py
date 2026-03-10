"""Validate Macenko stain separation on IHC ground-truth images.

Runs one or two reference modes and compares them:
  - "none"   : per-image Macenko, no batch reference.
  - "config" : reads stain params from a YAML config and runs with batch_median /
               batch_avg + ref_blend, matching training conditions exactly.

For each mode and every image:
  1. Estimate stain matrix V (3×2, columns = [H, DAB]).
  2. Deconvolve → H and DAB concentration maps.
  3. Reconstruct RGB using UNCLAMPED concentrations + I0 normalisation inverse.
  4. Compute MSE, PSNR, MAE vs original.

Outputs (per mode, under <output_dir>/<split>/<mode>/):
  metrics.csv            – per-image PSNR / MSE / MAE
  error_distribution.png – PSNR histogram + CDF, MSE / MAE histograms
  worst_grid.png         – grid of worst-K images (original / reconstructed)
  failed/<stem>_fail.png – 7-panel figure per failing image

When both modes are run, additionally writes to <output_dir>/<split>/:
  comparison.csv         – both modes + delta per image
  mode_comparison.png    – overlaid distributions + scatter + delta histogram
  side_by_side/          – worst images shown with both reconstructions

Usage
-----
    # none mode only
    python validate_stain_separation.py --split val

    # both modes (none + config)
    python validate_stain_separation.py --split val --config outputs/expr/run_001/config.yaml

    # override fail threshold and top-k
    python validate_stain_separation.py --split val --config outputs/expr/run_001/config.yaml \\
        --fail_psnr 35 --top_k 20
"""

import argparse
import csv
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from skimage import io
from skimage.exposure import rescale_intensity

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.stains.macenko import estimate_stain_matrix, rgb_to_od  # noqa: E402

EPS = 1e-6


# ── config loading ────────────────────────────────────────────────────────────

def load_stain_params(config_path: str) -> dict:
    """Extract stain-related params from a training YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    expr = cfg.get("expression", {})
    training = cfg.get("training", {})
    return {
        "reference_mode": expr.get("stain_reference_mode", "batch_median"),
        "ref_blend":      float(expr.get("stain_ref_blend", 0.0)),
        "min_ref_images": int(expr.get("stain_min_ref_images", 6)),
        "od_threshold":   float(expr.get("od_threshold", 0.15)),
        "batch_size":     int(training.get("batch_size", 8)),
    }


# ── image I/O ─────────────────────────────────────────────────────────────────

def load_rgb_tensor(path: Path) -> torch.Tensor:
    """Load PNG/JPG as float32 (3,H,W) in [0,1]."""
    img = io.imread(str(path)).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    if img.shape[2] == 4:
        img = img[..., :3]
    return torch.from_numpy(img.transpose(2, 0, 1))


def _resize_for_estimation(img: torch.Tensor, max_side: int) -> torch.Tensor:
    """Downscale (3,H,W) so the longer side ≤ max_side. No-op if already smaller."""
    _, H, W = img.shape
    if max(H, W) <= max_side:
        return img
    scale = max_side / max(H, W)
    new_h, new_w = max(1, int(H * scale)), max(1, int(W * scale))
    return F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                         mode="bilinear", align_corners=False).squeeze(0)


# ── core reconstruction (given a pre-estimated V) ─────────────────────────────

def _reconstruct(img_01: torch.Tensor, V: torch.Tensor) -> tuple:
    """
    Deconvolve and reconstruct an image given an already-estimated stain matrix V.

    Reconstruction uses UNCLAMPED concentrations + I0 correction:
      img = I0 × 10^(−(S_H·V[:,0] + S_DAB·V[:,1]))
    This accurately inverts rgb_to_od and avoids the phantom-stain artefact
    caused by clamping near-zero background concentrations to 0.

    Clamped h_map / dab_map are still returned for individual-stain visualisation.

    Returns
    -------
    h_map   : (H,W) clamped ≥0 H   concentration
    dab_map : (H,W) clamped ≥0 DAB concentration
    h_rgb   : (H,W,3) H   stain rendered to RGB
    dab_rgb : (H,W,3) DAB stain rendered to RGB
    recon   : (H,W,3) full reconstruction (unclamped, with I0)
    """
    H_px, W_px = img_01.shape[1], img_01.shape[2]

    # I0: per-channel 99th-percentile used inside rgb_to_od; needed to invert it
    I0 = torch.quantile(img_01.reshape(3, -1), 0.99, dim=1).clamp(min=EPS)  # (3,)

    od      = rgb_to_od(img_01)                   # (3,H,W)
    od_flat = od.reshape(3, -1)                   # (3,HW)
    S       = torch.linalg.pinv(V) @ od_flat      # (2,HW) raw, may be negative

    # Clamped for individual-stain rendering
    h_map   = S[0].reshape(H_px, W_px).clamp(min=0)
    dab_map = S[1].reshape(H_px, W_px).clamp(min=0)

    def _render_clamped(conc: torch.Tensor, col: int) -> np.ndarray:
        od_s = (conc.reshape(-1).unsqueeze(0) * V[:, col].unsqueeze(1)).reshape(3, H_px, W_px)
        return torch.pow(10.0, -od_s).clamp(0.0, 1.0).permute(1, 2, 0).numpy()

    h_rgb   = _render_clamped(h_map,   0)
    dab_rgb = _render_clamped(dab_map, 1)

    # Unclamped + I0 for accurate reconstruction
    od_recon = (S[0].unsqueeze(0) * V[:, 0].unsqueeze(1) +
                S[1].unsqueeze(0) * V[:, 1].unsqueeze(1)).reshape(3, H_px, W_px)
    recon = (torch.pow(10.0, -od_recon) * I0[:, None, None]).clamp(0.0, 1.0)
    recon = recon.permute(1, 2, 0).numpy()

    return h_map.numpy(), dab_map.numpy(), h_rgb, dab_rgb, recon


# ── quality metrics ───────────────────────────────────────────────────────────

def compute_metrics(original: np.ndarray, recon: np.ndarray) -> dict:
    diff = original.astype(np.float32) - recon.astype(np.float32)
    mse  = float(np.mean(diff ** 2))
    psnr = float(10.0 * math.log10(1.0 / mse)) if mse > EPS else float("inf")
    mae  = float(np.mean(np.abs(diff)))
    return {"mse": mse, "psnr": psnr, "mae": mae}


def _make_result(path, img_t, V):
    original = img_t.permute(1, 2, 0).numpy()
    h_map, dab_map, h_rgb, dab_rgb, recon = _reconstruct(img_t, V)
    return {
        "path": path, "original": original,
        "h_map": h_map, "dab_map": dab_map,
        "h_rgb": h_rgb, "dab_rgb": dab_rgb,
        "recon": recon, "V": V,
        **compute_metrics(original, recon),
        "error": None,
    }


# ── mode runners ──────────────────────────────────────────────────────────────

# resize is stored as a module-level var so pool workers can read it without args
_RESIZE_FOR_ESTIMATION: int = 0


def _process_one_none(path: Path) -> dict:
    """Worker for none mode (single-image, pool-safe)."""
    try:
        img_t  = load_rgb_tensor(path)
        img_sm = _resize_for_estimation(img_t, _RESIZE_FOR_ESTIMATION) if _RESIZE_FOR_ESTIMATION > 0 else img_t
        V      = estimate_stain_matrix(img_sm.unsqueeze(0), reference_mode="none")[0]
        return _make_result(path, img_t, V)   # reconstruct at full resolution
    except Exception as exc:
        return {"path": path, "error": str(exc),
                "mse": float("nan"), "psnr": float("nan"), "mae": float("nan")}


def run_none_mode(paths: list, n_workers: int, resize: int) -> list:
    """Process all paths with reference_mode='none' (parallel via Pool)."""
    global _RESIZE_FOR_ESTIMATION
    _RESIZE_FOR_ESTIMATION = resize
    tag_resize = f" (estimation resized to ≤{resize}px)" if resize > 0 else ""
    print(f"\n[none mode] Processing {len(paths)} images{tag_resize} …")
    t0 = time.time()
    results = []
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_process_one_none, paths), 1):
                results.append(r)
                tag = "ERR" if r["error"] else f"PSNR={r['psnr']:.1f}"
                elapsed = time.time() - t0
                print(f"  [none {i}/{len(paths)}] {r['path'].name}  {tag}  ({elapsed:.0f}s elapsed)")
    else:
        for i, p in enumerate(paths, 1):
            r = _process_one_none(p)
            results.append(r)
            tag = "ERR" if r["error"] else f"PSNR={r['psnr']:.1f}"
            elapsed = time.time() - t0
            print(f"  [none {i}/{len(paths)}] {p.name}  {tag}  ({elapsed:.0f}s elapsed)")
    return results


def run_config_mode(paths: list, stain_params: dict, resize: int) -> list:
    """
    Process all paths with the config stain params.

    batch_median / batch_avg reference modes need a batch to estimate V_ref,
    so images are grouped in batches matching the training batch_size.  Within
    each batch, estimate_stain_matrix is called once for all images together,
    exactly as during training.

    Stain matrix V is estimated from images resized to ≤ `resize` px (if > 0);
    reconstruction and metrics are always computed at the original resolution.
    1024×1024 images drop to 512×512 for estimation: ~4× fewer pixels → ~4× faster.
    """
    bsz       = stain_params["batch_size"]
    ref_mode  = stain_params["reference_mode"]
    ref_blend = stain_params["ref_blend"]
    min_ref   = stain_params["min_ref_images"]
    od_thr    = stain_params["od_threshold"]

    tag_resize = f", estimation resized to ≤{resize}px" if resize > 0 else ""
    print(f"\n[config mode: {ref_mode}, blend={ref_blend}{tag_resize}] "
          f"Processing {len(paths)} images in batches of {bsz} …")

    results = []
    t0 = time.time()
    n_batches = math.ceil(len(paths) / bsz)

    for batch_idx, batch_start in enumerate(range(0, len(paths), bsz), 1):
        batch_paths = paths[batch_start: batch_start + bsz]
        imgs = []       # (path, full_res_tensor)
        imgs_sm = []    # downscaled tensors for V estimation

        for p in batch_paths:
            try:
                t_full = load_rgb_tensor(p)
                t_sm   = _resize_for_estimation(t_full, resize) if resize > 0 else t_full
                imgs.append((p, t_full))
                imgs_sm.append(t_sm)
            except Exception as exc:
                results.append({"path": p, "error": str(exc),
                                 "mse": float("nan"), "psnr": float("nan"),
                                 "mae": float("nan")})

        if not imgs:
            continue

        elapsed = time.time() - t0
        print(f"  [config batch {batch_idx}/{n_batches}] "
              f"images {batch_start+1}–{batch_start+len(imgs)}/{len(paths)}  "
              f"estimating V … ({elapsed:.0f}s elapsed)", flush=True)

        # Estimate V from (optionally) downscaled batch
        batch_sm = torch.stack(imgs_sm)   # (B,3,h,w) – small
        try:
            V_list = estimate_stain_matrix(
                batch_sm,
                reference_mode=ref_mode,
                ref_blend=ref_blend,
                min_ref_images=min_ref,
                od_threshold=od_thr,
            )
        except Exception as exc:
            for p, _ in imgs:
                results.append({"path": p, "error": str(exc),
                                 "mse": float("nan"), "psnr": float("nan"),
                                 "mae": float("nan")})
            continue

        # Reconstruct at full resolution using the estimated V
        for i, ((p, img_t), V) in enumerate(zip(imgs, V_list)):
            try:
                r = _make_result(p, img_t, V)
            except Exception as exc:
                r = {"path": p, "error": str(exc),
                     "mse": float("nan"), "psnr": float("nan"), "mae": float("nan")}
            results.append(r)
            tag = "ERR" if r["error"] else f"PSNR={r['psnr']:.1f}"
            elapsed = time.time() - t0
            print(f"    [{batch_start+i+1}/{len(paths)}] {p.name}  {tag}  ({elapsed:.0f}s elapsed)")

    return results


# ── figure helpers ────────────────────────────────────────────────────────────

def save_failure_figure(result: dict, out_path: Path, mode_label: str = "") -> None:
    """7-panel figure: Original | H-map | DAB-map | H-rgb | DAB-rgb | Recon | Error."""
    original = result["original"]
    h_map    = result["h_map"]
    dab_map  = result["dab_map"]
    h_rgb    = result["h_rgb"]
    dab_rgb  = result["dab_rgb"]
    recon    = result["recon"]
    err_map  = np.abs(original - recon).mean(axis=2)

    fig, axes = plt.subplots(1, 7, figsize=(28, 4))
    panels = [
        (original,                               None,    None),
        (rescale_intensity(h_map,   out_range=(0,1)), "gray", None),
        (rescale_intensity(dab_map, out_range=(0,1)), "gray", None),
        (h_rgb,                                  None,    None),
        (dab_rgb,                                None,    None),
        (recon,                                  None,    None),
        (err_map,                                "hot",   (0, err_map.max() + EPS)),
    ]
    titles = ["Original IHC", "H map", "DAB map",
              "H rendered", "DAB rendered", "Reconstructed", "Abs error"]
    for ax, (img, cmap, clim), title in zip(axes, panels, titles):
        kw = {}
        if cmap:
            kw["cmap"] = cmap
        if clim:
            kw["vmin"], kw["vmax"] = clim
        im = ax.imshow(img, **kw)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if cmap == "hot":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

    stem = result["path"].stem
    tag  = f"  [{mode_label}]" if mode_label else ""
    fig.suptitle(
        f"{stem}{tag} | PSNR={result['psnr']:.2f} dB  MSE={result['mse']:.5f}  MAE={result['mae']:.5f}",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_side_by_side_figure(r_none: dict, r_cfg: dict, out_path: Path) -> None:
    """
    2-row figure comparing none vs config for one image.
    Row 0: Original | none H-map | none DAB-map | none Recon | none Error
    Row 1: Original | cfg  H-map | cfg  DAB-map | cfg  Recon | cfg  Error
    """
    original = r_none["original"]

    def _row(result, label):
        h_d  = rescale_intensity(result["h_map"],   out_range=(0, 1))
        d_d  = rescale_intensity(result["dab_map"], out_range=(0, 1))
        recon = result["recon"]
        err   = np.abs(original - recon).mean(axis=2)
        return [(original, None, f"Original [{label}]"),
                (h_d,      "gray", f"H map"),
                (d_d,      "gray", f"DAB map"),
                (recon,    None,   f"Recon  PSNR={result['psnr']:.1f}dB"),
                (err,      "hot",  f"Error  MAE={result['mae']:.4f}")]

    rows = [_row(r_none, "none"), _row(r_cfg, "config")]
    fig, axes = plt.subplots(2, 5, figsize=(22, 7))
    for row_idx, (row_data, result) in enumerate(zip(rows, [r_none, r_cfg])):
        for col_idx, (img, cmap, title) in enumerate(row_data):
            ax = axes[row_idx, col_idx]
            kw = {"cmap": cmap} if cmap else {}
            if cmap == "hot":
                kw["vmin"] = 0
                kw["vmax"] = np.abs(original - r_none["recon"]).mean(axis=2).max() + EPS
            ax.imshow(img, **kw)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(["none", "config"][row_idx],
                                    fontsize=10, fontweight="bold",
                                    rotation=0, labelpad=45, va="center")

    stem   = r_none["path"].stem
    delta  = r_cfg["psnr"] - r_none["psnr"]
    fig.suptitle(f"{stem} | ΔPSNR (config−none) = {delta:+.2f} dB", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_error_distribution(results: list, out_path: Path,
                             fail_psnr: float, label: str = "") -> None:
    """Histogram + CDF of PSNR, MSE, MAE for one mode."""
    psnr_vals = [r["psnr"] for r in results if math.isfinite(r.get("psnr", float("nan")))]
    mse_vals  = [r["mse"]  for r in results if math.isfinite(r.get("mse",  float("nan")))]
    mae_vals  = [r["mae"]  for r in results if math.isfinite(r.get("mae",  float("nan")))]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    title_prefix = f"[{label}] " if label else ""

    axes[0, 0].hist(psnr_vals, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0, 0].axvline(fail_psnr, color="red", linestyle="--", linewidth=1.5,
                        label=f"fail={fail_psnr} dB")
    axes[0, 0].set_xlabel("PSNR (dB)"); axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"{title_prefix}PSNR distribution  (n={len(psnr_vals)})"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    sorted_p   = np.sort(psnr_vals)
    fail_frac  = float(np.mean(np.array(psnr_vals) < fail_psnr)) * 100
    axes[0, 1].plot(sorted_p, np.linspace(0, 1, len(sorted_p)), color="steelblue", linewidth=2)
    axes[0, 1].axvline(fail_psnr, color="red", linestyle="--", linewidth=1.5)
    axes[0, 1].set_xlabel("PSNR (dB)"); axes[0, 1].set_ylabel("CDF")
    axes[0, 1].set_title(f"{title_prefix}PSNR CDF  ({fail_frac:.1f}% below threshold)"); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].hist(mse_vals, bins=60, color="coral",    edgecolor="white", alpha=0.85)
    axes[1, 0].set_xlabel("MSE"); axes[1, 0].set_ylabel("Count"); axes[1, 0].set_title(f"{title_prefix}MSE"); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(mae_vals, bins=60, color="seagreen", edgecolor="white", alpha=0.85)
    axes[1, 1].set_xlabel("MAE"); axes[1, 1].set_ylabel("Count"); axes[1, 1].set_title(f"{title_prefix}MAE"); axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Macenko reconstruction quality  {title_prefix}", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def save_worst_grid(worst: list, out_path: Path, ncols: int = 4) -> None:
    """Grid of worst-N images: original on top row, reconstructed below."""
    n = len(worst); ncols = min(ncols, n)
    nrows = math.ceil(n / ncols) * 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * (nrows // 2)))
    axes = np.array(axes).reshape(nrows, ncols)
    for idx, result in enumerate(worst):
        col = idx % ncols; tr = (idx // ncols) * 2; br = tr + 1
        axes[tr, col].imshow(result["original"]); axes[tr, col].axis("off")
        axes[tr, col].set_title(f"{result['path'].stem[:16]}\nPSNR={result['psnr']:.1f} dB", fontsize=8)
        axes[br, col].imshow(result["recon"]); axes[br, col].axis("off")
        axes[br, col].set_title(f"Recon  MAE={result['mae']:.4f}", fontsize=8)
    for idx in range(n, ncols * (nrows // 2)):
        col = idx % ncols
        for r in range((idx // ncols) * 2, (idx // ncols) * 2 + 2):
            if r < nrows: axes[r, col].set_axis_off()
    fig.suptitle(f"Worst {n} reconstructions (Original / Reconstructed)", fontsize=12)
    plt.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out_path}")


# ── comparison figures (both modes) ──────────────────────────────────────────

def save_mode_comparison_figure(r_none: list, r_cfg: list,
                                 out_path: Path, fail_psnr: float,
                                 cfg_label: str) -> None:
    """
    4-panel figure comparing none vs config mode across all images:
      [0,0] PSNR overlaid histograms
      [0,1] PSNR CDF overlaid
      [1,0] Per-image scatter: PSNR_none vs PSNR_config
      [1,1] Delta histogram: PSNR_config − PSNR_none
    """
    by_name_none = {r["path"].name: r for r in r_none if not r.get("error")}
    by_name_cfg  = {r["path"].name: r for r in r_cfg  if not r.get("error")}
    common = sorted(set(by_name_none) & set(by_name_cfg))

    psnr_none = np.array([by_name_none[n]["psnr"] for n in common])
    psnr_cfg  = np.array([by_name_cfg[n]["psnr"]  for n in common])
    delta     = psnr_cfg - psnr_none

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    bins = np.linspace(min(psnr_none.min(), psnr_cfg.min()),
                       max(psnr_none.max(), psnr_cfg.max()), 60)

    # overlaid histograms
    axes[0, 0].hist(psnr_none, bins=bins, alpha=0.6, color="steelblue", label="none",   density=True)
    axes[0, 0].hist(psnr_cfg,  bins=bins, alpha=0.6, color="coral",     label=cfg_label, density=True)
    axes[0, 0].axvline(fail_psnr, color="red", linestyle="--", linewidth=1.5, label=f"fail={fail_psnr}")
    axes[0, 0].set_xlabel("PSNR (dB)"); axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("PSNR distribution: none vs config"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # overlaid CDFs
    for arr, color, label in [(psnr_none, "steelblue", "none"), (psnr_cfg, "coral", cfg_label)]:
        s = np.sort(arr)
        axes[0, 1].plot(s, np.linspace(0, 1, len(s)), color=color, linewidth=2, label=label)
    axes[0, 1].axvline(fail_psnr, color="red", linestyle="--", linewidth=1.5)
    axes[0, 1].set_xlabel("PSNR (dB)"); axes[0, 1].set_ylabel("CDF")
    axes[0, 1].set_title("PSNR CDF: none vs config"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # per-image scatter
    improved = delta >= 0
    axes[1, 0].scatter(psnr_none[improved],  psnr_cfg[improved],
                       s=6, alpha=0.4, c="seagreen", label=f"config better ({improved.sum()})")
    axes[1, 0].scatter(psnr_none[~improved], psnr_cfg[~improved],
                       s=6, alpha=0.4, c="firebrick", label=f"none better ({(~improved).sum()})")
    lim = max(psnr_none.max(), psnr_cfg.max()) + 1
    lo  = min(psnr_none.min(), psnr_cfg.min()) - 1
    axes[1, 0].plot([lo, lim], [lo, lim], "k--", linewidth=1, alpha=0.7, label="Equal")
    axes[1, 0].set_xlim(lo, lim); axes[1, 0].set_ylim(lo, lim)
    axes[1, 0].set_xlabel("PSNR – none"); axes[1, 0].set_ylabel(f"PSNR – {cfg_label}")
    axes[1, 0].set_title("Per-image PSNR scatter"); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

    # delta histogram
    axes[1, 1].hist(delta, bins=60, color="mediumpurple", edgecolor="white", alpha=0.85)
    axes[1, 1].axvline(0, color="black", linewidth=1.2)
    axes[1, 1].axvline(delta.mean(), color="red", linestyle="--", linewidth=1.5,
                        label=f"mean Δ={delta.mean():+.2f} dB")
    axes[1, 1].set_xlabel(f"ΔPSNR ({cfg_label} − none)"); axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title(f"ΔPSNR distribution  ({(delta>0).mean()*100:.1f}% config wins)")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Mode comparison: none vs {cfg_label}  (n={len(common)} paired images)",
                 fontsize=13)
    plt.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out_path}")


def save_comparison_csv(r_none: list, r_cfg: list, out_path: Path, cfg_label: str) -> None:
    by_none = {r["path"].name: r for r in r_none}
    by_cfg  = {r["path"].name: r for r in r_cfg}
    all_names = sorted(set(by_none) | set(by_cfg))
    fields = ["filename",
              "psnr_none", "mse_none", "mae_none",
              f"psnr_{cfg_label}", f"mse_{cfg_label}", f"mae_{cfg_label}",
              "delta_psnr", "winner"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name in all_names:
            rn = by_none.get(name, {})
            rc = by_cfg.get(name, {})
            pn = rn.get("psnr", float("nan"))
            pc = rc.get("psnr", float("nan"))
            delt = pc - pn if (math.isfinite(pn) and math.isfinite(pc)) else float("nan")
            winner = ("config" if delt > 0 else "none") if math.isfinite(delt) else "n/a"
            w.writerow({
                "filename":         name,
                "psnr_none":        f"{pn:.4f}",
                "mse_none":         f"{rn.get('mse', float('nan')):.8f}",
                "mae_none":         f"{rn.get('mae', float('nan')):.8f}",
                f"psnr_{cfg_label}": f"{pc:.4f}",
                f"mse_{cfg_label}":  f"{rc.get('mse', float('nan')):.8f}",
                f"mae_{cfg_label}":  f"{rc.get('mae', float('nan')):.8f}",
                "delta_psnr":       f"{delt:.4f}",
                "winner":           winner,
            })
    print(f"Saved → {out_path}")


# ── per-mode output orchestration ─────────────────────────────────────────────

def _valid(results: list) -> list:
    return [r for r in results if not r.get("error") and math.isfinite(r.get("psnr", float("nan")))]


def write_mode_outputs(results: list, out_dir: Path,
                       fail_psnr: float, top_k: int, label: str) -> None:
    """Write CSV, distribution plot, worst grid, and failure figures for one mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "failed").mkdir(exist_ok=True)

    # CSV
    fields = ["filename", "psnr", "mse", "mae", "error"]
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({"filename": r["path"].name,
                        "psnr":  f"{r['psnr']:.6f}",
                        "mse":   f"{r['mse']:.8f}",
                        "mae":   f"{r['mae']:.8f}",
                        "error": r.get("error") or ""})
    print(f"Saved → {out_dir / 'metrics.csv'}")

    valid = sorted(_valid(results), key=lambda r: r["psnr"])

    if not valid:
        print(f"[{label}] No valid results.")
        return

    save_error_distribution(valid, out_dir / "error_distribution.png", fail_psnr, label=label)

    worst_k = valid[:top_k]
    save_worst_grid(worst_k, out_dir / "worst_grid.png")

    # Failure figures
    failed_thresh = [r for r in valid if r["psnr"] < fail_psnr]
    to_save = list({r["path"]: r for r in failed_thresh + worst_k}.values())
    to_save.sort(key=lambda r: r["psnr"])
    for r in to_save:
        save_failure_figure(r, out_dir / "failed" / f"{r['path'].stem}_fail.png", mode_label=label)

    # Errors log
    errored = [r for r in results if r.get("error")]
    if errored:
        with open(out_dir / "errors.txt", "w") as f:
            for r in errored: f.write(f"{r['path'].name}: {r['error']}\n")

    arr = np.array([r["psnr"] for r in valid])
    print(f"\n[{label}] PSNR mean={arr.mean():.2f}  median={np.median(arr):.2f}"
          f"  min={arr.min():.2f}  max={arr.max():.2f} dB")
    print(f"[{label}] Failed (PSNR < {fail_psnr}): {len(failed_thresh)}/{len(valid)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate Macenko stain separation via RGB reconstruction."
    )
    parser.add_argument("--split",      default="val", choices=["train", "val"])
    parser.add_argument("--config",     default=None,
                        help="Path to training YAML config.  When provided, also runs the "
                             "config stain mode and produces a comparison.")
    parser.add_argument("--fail_psnr",  type=float, default=35.0,
                        help="PSNR threshold (dB) for 'failed' (default: 35)")
    parser.add_argument("--top_k",      type=int,   default=16)
    parser.add_argument("--output_dir", default="outputs/stain_validation")
    parser.add_argument("--n_workers",  type=int,   default=1,
                        help="Workers for none-mode Pool (default: 1; config mode is always sequential)")
    parser.add_argument("--resize",     type=int,   default=512,
                        help="Resize images to ≤ this side length before Macenko V estimation "
                             "(reconstruction/metrics always use full resolution). "
                             "Set 0 to disable. Default: 512 — matches training crop size and "
                             "avoids slow SVD on 1024×1024 full-resolution images.")
    args = parser.parse_args()

    ihc_dir = ROOT / "data" / "raw" / args.split / "IHC"
    if not ihc_dir.exists():
        raise FileNotFoundError(f"IHC directory not found: {ihc_dir}")
    paths = sorted(ihc_dir.glob("*.png"))
    print(f"Found {len(paths)} IHC images in {ihc_dir}")

    base_dir = Path(args.output_dir) / args.split

    if args.resize > 0:
        print(f"Stain estimation will use images resized to ≤{args.resize}px "
              f"(reconstruction/metrics at full resolution).")

    # ── none mode ─────────────────────────────────────────────────────────────
    r_none = run_none_mode(paths, args.n_workers, args.resize)
    write_mode_outputs(r_none, base_dir / "none", args.fail_psnr, args.top_k, "none")

    # ── config mode (optional) ────────────────────────────────────────────────
    if args.config:
        stain_params = load_stain_params(args.config)
        cfg_label = stain_params["reference_mode"]
        print(f"\nConfig stain params: {stain_params}")

        r_cfg = run_config_mode(paths, stain_params, args.resize)
        write_mode_outputs(r_cfg, base_dir / cfg_label, args.fail_psnr, args.top_k, cfg_label)

        # ── comparison outputs ────────────────────────────────────────────────
        print("\nBuilding comparison outputs …")
        save_comparison_csv(r_none, r_cfg, base_dir / "comparison.csv", cfg_label)
        save_mode_comparison_figure(r_none, r_cfg, base_dir / "mode_comparison.png",
                                    args.fail_psnr, cfg_label)

        # Side-by-side panels for images that differ the most between modes
        by_none = {r["path"].name: r for r in _valid(r_none)}
        by_cfg  = {r["path"].name: r for r in _valid(r_cfg)}
        common  = [(name, by_cfg[name]["psnr"] - by_none[name]["psnr"])
                   for name in set(by_none) & set(by_cfg)]
        common.sort(key=lambda x: abs(x[1]), reverse=True)   # biggest delta first

        sbs_dir = base_dir / "side_by_side"
        sbs_dir.mkdir(exist_ok=True)
        for name, delta in common[:args.top_k]:
            out_png = sbs_dir / f"{Path(name).stem}_delta{delta:+.1f}.png"
            save_side_by_side_figure(by_none[name], by_cfg[name], out_png)
            print(f"  SBS → {out_png.name}  Δ={delta:+.2f} dB")

        print(f"\nAll outputs in: {base_dir}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
