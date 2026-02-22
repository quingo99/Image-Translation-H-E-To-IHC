"""Phase 2 evaluation script: inference on test set, writes metrics CSV.

Usage:
    python eval.py --config configs/base.yaml --checkpoint outputs/base/run_001/generator_best.pth
    python eval.py --config configs/expr.yaml --checkpoint outputs/expr/run_001/generator_best.pth

Produces: outputs/<model>/run_NNN/metrics_<split>.csv
Ground truth: test/IHC or data/raw/groundtruth/; falls back to val split.
"""

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.bci_dataset import BCIDataset
from src.metrics.metrics import (
    compute_dab_metrics,
    compute_lpips,
    compute_membrane_intensity_error,
    compute_nuclei_density_error,
    compute_psnr,
    compute_ssim,
    load_lpips_model,
)
from src.models.pix2pix import Generator
from src.utils.repro import seed_everything

matplotlib.use("Agg")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_sample_triplet(he, real, fake, fname, path):
    """Save one sample as a 3-panel image: H&E, real IHC, generated IHC."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for j, (img, title) in enumerate(
        [(he, "H&E Input"), (real, "Real IHC"), (fake, "Generated IHC")]
    ):
        arr = ((img.cpu().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        axes[j].imshow(arr)
        axes[j].set_title(title)
        axes[j].axis("off")
    fig.suptitle(str(fname))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate(cfg, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg["seed"])

    # Output dir: same parent as checkpoint
    out_dir = str(Path(checkpoint_path).parent)
    eval_dir = os.path.join(out_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    sample_dir = os.path.join(eval_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # Load model
    G = Generator(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        norm_type=cfg["model"].get("norm", "instance"),
    ).to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()

    data_cfg = cfg["data"]
    eval_full_resolution = bool(data_cfg.get("eval_full_resolution", False))
    eval_batch_size_default = (
        1 if eval_full_resolution else cfg["training"]["batch_size"]
    )
    eval_batch_size = int(
        cfg["training"].get("eval_batch_size", eval_batch_size_default)
    )
    if eval_batch_size < 1:
        raise ValueError("training.eval_batch_size must be >= 1")

    # Evaluation data: try test first, then val fallback
    eval_split = "test"
    try:
        test_ds = BCIDataset(
            data_cfg["root_dir"],
            split="test",
            image_size=data_cfg["image_size"],
            use_full_resolution=eval_full_resolution,
        )
    except (RuntimeError, FileNotFoundError):
        print("Test split has no paired IHC data. Using val split for evaluation.")
        eval_split = "val"
        test_ds = BCIDataset(
            data_cfg["root_dir"],
            split="val",
            image_size=data_cfg["image_size"],
            use_full_resolution=eval_full_resolution,
        )

    print(f"Evaluation split: {eval_split} ({len(test_ds)} samples)")
    if eval_full_resolution:
        print(f"Evaluation transform: full resolution (batch_size={eval_batch_size})")
    else:
        print(
            "Evaluation transform: center crop "
            f"{data_cfg['image_size']}x{data_cfg['image_size']} "
            f"(batch_size={eval_batch_size})"
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    lpips_model = load_lpips_model(device)
    if lpips_model is None:
        print("WARNING: `lpips` is not installed. LPIPS values will be NaN.")

    # Inference and metrics
    print(f"Running inference on {eval_split} set...")
    all_rows = []
    batch_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            x = batch["he"].to(device)
            y = batch["ihc"].to(device)
            fnames = batch["filename"]

            y_hat = G(x)

            # 9.1 Image similarity
            psnr_vals = compute_psnr(y, y_hat)
            ssim_vals = compute_ssim(y, y_hat)
            lpips_vals = compute_lpips(y, y_hat, lpips_model)

            # 9.2 DAB expression
            dab_metrics = compute_dab_metrics(y, y_hat)

            # 9.3 Structure
            nuclei_errs = compute_nuclei_density_error(y, y_hat)
            membrane_errs = compute_membrane_intensity_error(y, y_hat)

            for i in range(len(fnames)):
                row = {
                    "filename": fnames[i],
                    "psnr": psnr_vals[i],
                    "ssim": ssim_vals[i],
                    "lpips": lpips_vals[i],
                    **dab_metrics[i],
                    "nuclei_density_error": nuclei_errs[i],
                    "membrane_intensity_error": membrane_errs[i],
                }
                all_rows.append(row)

            if batch_idx < 10:
                for i, fname in enumerate(fnames):
                    stem = Path(fname).stem
                    save_sample_triplet(
                        x[i],
                        y[i],
                        y_hat[i],
                        fname,
                        os.path.join(sample_dir, f"{batch_idx:03d}_{i:02d}_{stem}.png"),
                    )

            batch_idx += 1

    # Save CSV
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, f"metrics_{eval_split}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to: {csv_path}")

    # Print summary
    print(f"\n=== {eval_split.upper()} Set Summary ===")
    for col in [
        "psnr",
        "ssim",
        "lpips",
        "dab_pearson_r",
        "iod_rel_err",
        "miod_rel_err",
        "nuclei_density_error",
        "membrane_intensity_error",
    ]:
        if col in df.columns:
            print(f"  {col}: {df[col].mean():.4f} +/- {df[col].std():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint)
