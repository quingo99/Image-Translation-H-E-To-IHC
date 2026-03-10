"""Inspect IOD/mIOD signal on IHC images and save debug visualizations.

Examples:
    python scripts/inspect_expression_signal.py --input data/BCI/train/IHC --recursive
    python scripts/inspect_expression_signal.py --input data/BCI/train/IHC/00001.png --show
"""

import argparse
import csv
from pathlib import Path
import sys

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running as a standalone script from repo root or subfolders.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stains.macenko import EPS, estimate_stain_matrix, get_dab_map, tissue_mask_from_od


VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute IOD/mIOD for IHC images and visualize signal quality."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Image file, directory, or glob pattern (e.g., data/BCI/train/IHC/*.png).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan when --input is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/signal_debug",
        help="Directory to save debug figures and summary CSV.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit number of images to process (0 means all).",
    )
    parser.add_argument(
        "--max-figures",
        type=int,
        default=20,
        help="Save debug figures for only the first N images (0 disables figure saving).",
    )
    parser.add_argument(
        "--min-iod-signal",
        type=float,
        default=500.0,
        help="Threshold used to flag low IOD.",
    )
    parser.add_argument(
        "--min-miod-signal",
        type=float,
        default=0.05,
        help="Threshold used to flag low mIOD.",
    )
    parser.add_argument(
        "--od-threshold",
        type=float,
        default=0.2,
        help="OD threshold for non-LV3 images.",
    )
    parser.add_argument(
        "--dilate-radius",
        type=int,
        default=2,
        help="Dilation radius for non-LV3 images.",
    )
    parser.add_argument(
        "--lv3-tag",
        type=str,
        default="3+",
        help="Filename tag used to identify LV3 images.",
    )
    parser.add_argument(
        "--od-threshold-lv3",
        type=float,
        default=0.25,
        help="OD threshold for LV3 images.",
    )
    parser.add_argument(
        "--dilate-radius-lv3",
        type=int,
        default=0,
        help="Dilation radius for LV3 images.",
    )
    parser.add_argument(
        "--dab-nonnegative",
        type=str,
        default="softplus",
        choices=["clamp", "softplus", "none"],
        help="Nonnegative mode for DAB map.",
    )
    parser.add_argument(
        "--softplus-beta",
        type=float,
        default=8.0,
        help="Softplus beta when --dab-nonnegative=softplus.",
    )
    parser.add_argument(
        "--stain-reference-mode",
        type=str,
        default="none",
        choices=["none", "batch_avg", "provided"],
        help="Reference mode passed to Macenko stain estimation.",
    )
    parser.add_argument(
        "--stain-ref-blend",
        type=float,
        default=0.0,
        help="Blend ratio with reference stain matrix.",
    )
    parser.add_argument(
        "--stain-min-ref-images",
        type=int,
        default=4,
        help="Minimum valid images for batch reference in batch_avg mode.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each figure in an interactive window.",
    )
    return parser.parse_args()


def collect_image_paths(input_arg, recursive=False):
    p = Path(input_arg)
    if p.exists():
        if p.is_file():
            return [p]
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            items = [x for x in p.glob(pattern) if x.is_file() and x.suffix.lower() in VALID_EXTS]
            return sorted(items)
    glob_items = [x for x in Path().glob(input_arg) if x.is_file() and x.suffix.lower() in VALID_EXTS]
    return sorted(glob_items)


def load_rgb_01(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_01 = rgb.astype(np.float32) / 255.0
    return rgb, rgb_01


def compute_iod_miod_single(rgb_01, args, od_threshold, dilate_radius):
    y_01 = torch.from_numpy(rgb_01).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        mask = tissue_mask_from_od(y_01, od_threshold, dilate_radius)[0]
        V = estimate_stain_matrix(
            y_01,
            od_threshold=od_threshold,
            reference_mode=args.stain_reference_mode,
            ref_blend=args.stain_ref_blend,
            min_ref_images=args.stain_min_ref_images,
        )[0]
        dab = get_dab_map(
            y_01[0],
            V,
            nonnegative=args.dab_nonnegative,
            softplus_beta=args.softplus_beta,
        )
        masked_dab = mask * dab
        tissue_area = mask.sum() + EPS
        iod = masked_dab.sum()
        miod = iod / tissue_area
    return mask.cpu().numpy(), dab.cpu().numpy(), float(iod.item()), float(miod.item())


def sanitize_name(path):
    parts = list(path.parts)
    if len(parts) > 4:
        parts = parts[-4:]
    return "__".join(parts).replace(":", "_")


def save_debug_figure(path, rgb, mask, dab, iod, miod, args, out_dir, od_threshold, dilate_radius, is_lv3):
    low_iod = iod < args.min_iod_signal
    low_miod = miod < args.min_miod_signal
    status_bits = []
    if low_iod:
        status_bits.append("LOW_IOD")
    if low_miod:
        status_bits.append("LOW_MIOD")
    status_text = " | ".join(status_bits) if status_bits else "OK"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("IHC RGB")
    axes[0].axis("off")

    axes[1].imshow(dab, cmap="magma")
    axes[1].set_title("DAB Map")
    axes[1].axis("off")

    im = axes[1].images[-1]
    axes[2].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Tissue Mask")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    title_color = "red" if (low_iod or low_miod) else "green"
    fig.suptitle(
        f"{path.name} | IOD={iod:.4f} (min {args.min_iod_signal}) | "
        f"mIOD={miod:.6f} (min {args.min_miod_signal}) | {status_text}\n"
        f"rule={'LV3' if is_lv3 else 'default'} | od={od_threshold} | dilate={dilate_radius}",
        color=title_color,
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / f"{sanitize_name(path)}.png"
    fig.savefig(out_path, dpi=140)
    if args.show:
        plt.show()
    plt.close(fig)
    return out_path, low_iod, low_miod


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.od_threshold <= 0 or args.od_threshold_lv3 <= 0:
        raise ValueError("OD thresholds must be > 0.")
    if args.dilate_radius < 0 or args.dilate_radius_lv3 < 0:
        raise ValueError("Dilation radii must be >= 0.")
    print(
        "Mask rules: "
        f"default(od={args.od_threshold}, dilate={args.dilate_radius}), "
        f"LV3 tag='{args.lv3_tag}' -> "
        f"(od={args.od_threshold_lv3}, dilate={args.dilate_radius_lv3})"
    )

    paths = collect_image_paths(args.input, recursive=args.recursive)
    if args.max_images and args.max_images > 0:
        paths = paths[: args.max_images]

    if not paths:
        raise FileNotFoundError("No valid image files found for --input.")

    rows = []
    saved_figures = 0
    for idx, path in enumerate(paths, start=1):
        rgb, rgb_01 = load_rgb_01(path)
        is_lv3 = bool(args.lv3_tag) and (args.lv3_tag in path.name)
        od_threshold = args.od_threshold_lv3 if is_lv3 else args.od_threshold
        dilate_radius = args.dilate_radius_lv3 if is_lv3 else args.dilate_radius
        mask, dab, iod, miod = compute_iod_miod_single(
            rgb_01, args, od_threshold, dilate_radius
        )
        low_iod = iod < args.min_iod_signal
        low_miod = miod < args.min_miod_signal
        fig_path = ""
        if args.max_figures > 0 and idx <= args.max_figures:
            fig_path, _, _ = save_debug_figure(
                path,
                rgb,
                mask,
                dab,
                iod,
                miod,
                args,
                out_dir,
                od_threshold,
                dilate_radius,
                is_lv3,
            )
            saved_figures += 1

        rows.append(
            {
                "image_path": str(path),
                "iod": iod,
                "miod": miod,
                "is_lv3": int(is_lv3),
                "od_threshold_used": float(od_threshold),
                "dilate_radius_used": int(dilate_radius),
                "low_iod": int(low_iod),
                "low_miod": int(low_miod),
                "low_both": int(low_iod and low_miod),
                "figure_path": str(fig_path),
            }
        )
        print(
            f"[{idx}/{len(paths)}] {path.name} | IOD={iod:.4f} | mIOD={miod:.6f} | "
            f"low_iod={int(low_iod)} low_miod={int(low_miod)}"
        )

    csv_path = out_dir / "signal_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "iod",
                "miod",
                "is_lv3",
                "od_threshold_used",
                "dilate_radius_used",
                "low_iod",
                "low_miod",
                "low_both",
                "figure_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    n_low_iod = sum(r["low_iod"] for r in rows)
    n_low_miod = sum(r["low_miod"] for r in rows)
    n_low_both = sum(r["low_both"] for r in rows)
    print("\nSummary")
    print(f"  images: {n}")
    print(f"  below min_iod_signal: {n_low_iod} ({n_low_iod / n:.2%})")
    print(f"  below min_miod_signal: {n_low_miod} ({n_low_miod / n:.2%})")
    print(f"  below both: {n_low_both} ({n_low_both / n:.2%})")
    print(f"  figures_saved: {saved_figures}")
    print(f"  csv: {csv_path}")
    print(f"  figures: {out_dir}")


if __name__ == "__main__":
    main()
