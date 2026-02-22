"""Visualize DAB extraction from IHC images using Macenko stain separation.

Usage:
    python test_dab.py
    python test_dab.py --input data/raw/val/IHC/00026_train_3+.png
    python test_dab.py --input_dir data/raw/val/IHC --n 8
    python test_dab.py --output my_dab_output.png
"""

import argparse
import os
import glob

import cv2
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.stains.macenko import estimate_stain_matrix, get_dab_map, tissue_mask_from_od


def load_image_as_tensor(path):
    """Load a single image file and return a (1, 3, H, W) tensor in [0, 1]."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).float() / 255.0  # [0, 1]
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)    # (1, 3, H, W)
    return tensor


def process_single(img_01, od_threshold=0.15, dilate_radius=3):
    """Run Macenko on a single (1, 3, H, W) tensor. Returns numpy arrays."""
    V_list = estimate_stain_matrix(img_01, od_threshold)
    V = V_list[0]

    dab = get_dab_map(img_01[0], V)  # (H, W)
    mask = tissue_mask_from_od(img_01, od_threshold, dilate_radius)[0]  # (H, W)

    return dab.cpu().numpy(), mask.cpu().numpy(), V.cpu().numpy()


def visualize_single(img_path, od_threshold=0.15, dilate_radius=3):
    """Process one image and return (rgb, dab, mask, V)."""
    img_01 = load_image_as_tensor(img_path)
    rgb = img_01[0].permute(1, 2, 0).numpy()  # (H, W, 3)
    dab, mask, V = process_single(img_01, od_threshold, dilate_radius)
    return rgb, dab, mask, V


def plot_single(img_path, output_path, od_threshold=0.15, dilate_radius=3):
    """Plot a single IHC image with its DAB map, tissue mask, and masked DAB."""
    rgb, dab, mask, V = visualize_single(img_path, od_threshold, dilate_radius)
    fname = os.path.basename(img_path)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(rgb)
    axes[0].set_title(f"IHC Input\n{fname}")

    im1 = axes[1].imshow(dab, cmap="hot", vmin=0)
    axes[1].set_title("DAB Concentration")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Tissue Mask")

    masked_dab = mask * dab
    im3 = axes[3].imshow(masked_dab, cmap="hot", vmin=0)
    axes[3].set_title("DAB on Tissue")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"Macenko Stain Separation — {fname}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_grid(img_paths, output_path, od_threshold=0.15, dilate_radius=3):
    """Plot a grid of N images, each showing IHC | DAB | Tissue Mask | Masked DAB."""
    n = len(img_paths)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[None, :]

    for row, img_path in enumerate(img_paths):
        rgb, dab, mask, V = visualize_single(img_path, od_threshold, dilate_radius)
        fname = os.path.basename(img_path)

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"IHC: {fname}", fontsize=9)

        im1 = axes[row, 1].imshow(dab, cmap="hot", vmin=0)
        axes[row, 1].set_title("DAB Concentration", fontsize=9)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        axes[row, 2].imshow(mask, cmap="gray")
        axes[row, 2].set_title("Tissue Mask", fontsize=9)

        masked_dab = mask * dab
        im3 = axes[row, 3].imshow(masked_dab, cmap="hot", vmin=0)
        axes[row, 3].set_title("DAB on Tissue", fontsize=9)
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)

        for col in range(4):
            axes[row, col].axis("off")

        # Print per-image stats
        tissue_area = mask.sum()
        iod = (mask * dab).sum()
        miod = iod / (tissue_area + 1e-8)
        print(f"  [{fname}] tissue_area={tissue_area:.0f}  IOD={iod:.2f}  mIOD={miod:.4f}")

    plt.suptitle("Macenko DAB Extraction from IHC Images", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DAB extraction from IHC via Macenko")
    parser.add_argument("--input", type=str, default=None, help="Path to a single IHC image")
    parser.add_argument("--input_dir", type=str, default="data/raw/val/IHC",
                        help="Directory of IHC images (used when --input is not set)")
    parser.add_argument("--n", type=int, default=4, help="Number of images to sample from input_dir")
    parser.add_argument("--output", type=str, default="dab_test_output.png", help="Output image path")
    parser.add_argument("--od_threshold", type=float, default=0.15, help="OD threshold for tissue mask")
    parser.add_argument("--dilate_radius", type=int, default=3, help="Dilation radius for tissue mask")
    args = parser.parse_args()

    if args.input:
        print(f"Processing single image: {args.input}")
        plot_single(args.input, args.output, args.od_threshold, args.dilate_radius)
    else:
        # Sample N images from the directory, picking diverse HER2 grades if possible
        all_imgs = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
        if not all_imgs:
            raise FileNotFoundError(f"No .png images found in {args.input_dir}")

        # Try to pick one of each grade: 0, 1+, 2+, 3+
        by_grade = {"0": [], "1+": [], "2+": [], "3+": []}
        for p in all_imgs:
            stem = os.path.basename(p).rsplit(".", 1)[0]
            grade = stem.split("_")[-1]
            if grade in by_grade:
                by_grade[grade].append(p)

        selected = []
        for grade in ["0", "1+", "2+", "3+"]:
            if by_grade[grade]:
                selected.append(by_grade[grade][0])
            if len(selected) >= args.n:
                break

        # Fill remaining slots if needed
        remaining = args.n - len(selected)
        if remaining > 0:
            extras = [p for p in all_imgs if p not in selected]
            selected.extend(extras[:remaining])

        selected = selected[:args.n]
        print(f"Processing {len(selected)} images from {args.input_dir}")
        plot_grid(selected, args.output, args.od_threshold, args.dilate_radius)


if __name__ == "__main__":
    main()
