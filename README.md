# Phase 2: Pyramid Pix2Pix with Clinically Guided Expression Loss

Virtual HER2 IHC staining from H&E using a conditional GAN with multi-scale Gaussian pyramid supervision and an optional DAB expression loss via Macenko stain separation.

Two models are trained and compared:

| Model | Description |
|---|---|
| `M_base` | Conditional GAN with adversarial + Gaussian pyramid L1 loss |
| `M_expr` | `M_base` + clinically guided expression loss `L_expr` |

---

## Project Structure

```
CS7640-Project/
├── configs/
│   ├── base.yaml            # M_base config (expression disabled)
│   └── expr.yaml            # M_expr config (expression enabled)
├── src/
│   ├── data/
│   │   └── bci_dataset.py   # Paired HE/IHC dataset
│   ├── losses/
│   │   ├── pyramid.py       # Gaussian pyramid L1 loss
│   │   └── expression.py    # Clinically guided DAB expression loss
│   ├── metrics/
│   │   └── metrics.py       # PSNR, SSIM, LPIPS, DAB, structure metrics
│   ├── models/
│   │   └── pix2pix.py       # U-Net generator + PatchGAN discriminator
│   ├── stains/
│   │   └── macenko.py       # Macenko stain separation + tissue masking
│   └── utils/
│       └── repro.py         # Seeding, run metadata, run directory helpers
├── scripts/
│   └── inspect_expression_signal.py  # DAB signal diagnostic tool
├── train.py                 # Training script
├── eval.py                  # Evaluation / inference script
├── report.py                # Comparative report generator
├── run.py                   # Unified entry point
├── colab_run.ipynb          # Google Colab training notebook
└── requirements.txt
```

**Outputs** (gitignored):
```
outputs/
├── base/run_001/            # M_base checkpoints, logs, samples
├── expr/run_001/            # M_expr checkpoints, logs, samples
└── report/                  # Comparison plots and tables
```

**Data** (gitignored):
```
data/raw/
├── train/HE/  + IHC/        # 3396 paired training images
├── val/HE/    + IHC/        # 500 paired validation images
├── test/HE/                 # 977 test images (no paired IHC)
└── groundtruth/             # Optional test IHC for evaluation fallback
```

---

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Place data under `data/raw/` with matching filenames across HE and IHC directories.

---

## Running

### Full pipeline (train both → evaluate both → report)

```bash
python run.py all
```

### Individual steps

```bash
# Train
python train.py --config configs/base.yaml
python train.py --config configs/expr.yaml

# Evaluate (auto-selects test split; falls back to val if no paired IHC)
python eval.py --config configs/base.yaml --checkpoint outputs/base/run_001/generator_best.pth
python eval.py --config configs/expr.yaml --checkpoint outputs/expr/run_001/generator_best.pth

# Generate comparison report
python report.py --base outputs/base/run_001/metrics_test.csv --expr outputs/expr/run_001/metrics_test.csv
```

### Colab

Open `colab_run.ipynb` in Google Colab. It expects a zip of this project
(with data inside) uploaded to Google Drive, then runs training + evaluation
automatically.

---

## Checkpoints

`train.py` saves several checkpoint files per run under `outputs/<model>/run_NNN/`:

| File | Saved when |
|---|---|
| `generator_best.pth` | Best value of `training.best_metric` |
| `generator_best_psnr.pth` | Best PSNR (when not the primary metric) |
| `generator_best_lpips.pth` | Best LPIPS (when not the primary metric) |
| `generator_best_expr.pth` | Best expression score (when not the primary metric) |
| `generator_final.pth` | End of training |
| `checkpoint_epoch_NNN.pth` | Every `training.save_every` epochs (full state for resuming) |

To resume training from a periodic checkpoint:
```yaml
# in your config:
training:
  resume: "outputs/expr/run_001/checkpoint_epoch_050.pth"
```

---

## Evaluation Outputs

`eval.py` writes under the same run directory as the checkpoint:

```
outputs/<model>/run_NNN/
└── evaluation/
    ├── metrics_val.csv        # Per-image metrics (val split)
    ├── metrics_test.csv       # Per-image metrics (test split, if IHC available)
    └── samples/               # Triplet visualizations (H&E | real IHC | generated IHC)
```

Metrics columns in the CSV:

| Column | Description |
|---|---|
| `psnr` | Peak signal-to-noise ratio (dB, higher is better) |
| `ssim` | Structural similarity (higher is better) |
| `lpips` | Perceptual distance via AlexNet (lower is better) |
| `iod_rel_err` | IOD relative error (DAB total, lower is better) |
| `miod_rel_err` | mIOD relative error (DAB mean per tissue area, lower is better) |
| `nuclei_density_error` | Hematoxylin area fraction error (lower is better) |
| `membrane_intensity_error` | Sobel edge energy error on DAB map (lower is better) |

---

## Configuration Reference

All parameters shared between `base.yaml` and `expr.yaml` unless noted.

### `data`

| Key | Default | Description |
|---|---|---|
| `root_dir` | `"data/raw"` | Root directory containing `train/`, `val/`, `test/` splits |
| `image_size` | `256` | Spatial size for training crops and val center crop |
| `val_image_size` | `512` | Override crop size for validation (defaults to `image_size`) |
| `train_crops_per_image` | `1` | Random crops drawn per training image per epoch |
| `train_crop_mode` | `"random"` | Crop strategy: `"random"` or `"quadrant_jitter"` |
| `val_full_resolution` | `false` | Skip center crop during validation; use original resolution |
| `eval_full_resolution` | `true` | Skip center crop during `eval.py`; use original resolution |
| `num_workers` | `4` | DataLoader worker processes |

### `model`

| Key | Default | Description |
|---|---|---|
| `num_res_blocks` | `6` | Residual blocks in the generator bottleneck |
| `in_channels` | `3` | Generator input channels (HE = RGB) |
| `out_channels` | `3` | Generator output channels (IHC = RGB) |
| `norm_G` | `"instance"` | Generator normalization: `"batch"`, `"instance"`, `"none"` |
| `norm_D` | `"none"` | Discriminator normalization |
| `spectral_norm_D` | `true` | Apply spectral normalization to all discriminator convolutions |

### `training`

| Key | Default | Description |
|---|---|---|
| `batch_size` | `8` | Training batch size |
| `val_batch_size` | `1` | Validation batch size (use `1` with `val_full_resolution`) |
| `eval_batch_size` | `1` | Eval batch size (use `1` with `eval_full_resolution`) |
| `epochs` | `100` | Total training epochs |
| `lr_G` | `0.0002` | Generator learning rate (Adam) |
| `lr_D` | `0.0002` | Discriminator learning rate (Adam) |
| `beta1` | `0.5` | Adam β₁ |
| `beta2` | `0.999` | Adam β₂ |
| `lambda_adv` | `5` | Adversarial loss weight |
| `lambda_pyr` | `25` | Gaussian pyramid L1 loss weight |
| `pyramid_levels` | `3` | Number of pyramid scales |
| `d_update_every` | `2` | Update discriminator every N generator steps |
| `gan_label_real` | `1.0` | Discriminator real target label |
| `gan_label_fake` | `0.0` | Discriminator fake target label |
| `gan_label_gen` | `1.0` | Generator adversarial target label |
| `val_schedule` | `"fixed"` | Validation trigger mode (only `"fixed"` supported) |
| `val_start_epoch` | `45` | First epoch eligible for validation |
| `val_every` | `5` | Run validation every N epochs after `val_start_epoch` |
| `val_tiling_mode` | `"none"` | Validation tiling: `"none"` or `"2x2"` |
| `val_tile_size` | `512` | Tile size for `"2x2"` tiling mode |
| `val_sample_n` | `4` | Number of sample triplets saved during validation |
| `best_metric` | `"lpips"` / `"expr"` | Metric for `generator_best.pth` selection |
| `save_extra_best_checkpoints` | `true` | Also save best-per-metric files for PSNR, LPIPS, expr |
| `save_every` | `10` | Save full checkpoint every N epochs |
| `checkpoint_dir` | `"outputs/base"` | Base directory for run subdirectories |
| `resume` | `null` | Path to a `checkpoint_epoch_NNN.pth` to resume from |

### `expression`

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` / `true` | Enable the expression loss (`M_expr` sets this to `true`) |
| `lambda_expr` | `2.2` | Expression loss weight |
| `epoch_add_expr` | `20` | Epoch at which expression loss is added to the total loss |
| `warmup_epochs` | `0` | Linearly ramp expression weight over this many epochs from `epoch_add_expr` |
| `od_threshold` | `0.15` | OD magnitude threshold for tissue vs. background separation |
| `dilate_radius` | `2` | Tissue mask dilation radius (pixels) for registration tolerance |
| `min_iod_signal` | `200` | Skip global IOD term if real IOD falls below this value |
| `min_miod_signal` | `0.001` | Skip global mIOD term if real mIOD falls below this value |
| `spatial_weight` | `0.4` | Weight of patch-wise spatial loss relative to global terms |
| `spatial_grid_size` | `8` | Grid resolution for patch-wise mIOD pooling (8 → 8×8 patches) |
| `spatial_min_tissue_frac` | `0.1` | Skip patches with tissue fraction below this threshold |
| `spatial_min_signal` | `0.001` | Skip patches with real mIOD below this threshold |
| `max_rel_term` | `5.0` | Clamp individual relative error terms to this maximum |
| `dab_nonnegative` | `"softplus"` | DAB nonnegativity mode: `"clamp"`, `"softplus"`, or `"none"` |
| `softplus_beta` | `8.0` | Softplus sharpness (higher → closer to hard clamp) |
| `stain_reference_mode` | `"batch_median"` | Stain reference aggregation: `"none"`, `"batch_avg"`, `"batch_median"` |
| `stain_ref_blend` | `0.25` | Blend ratio between per-image and batch-reference stain matrices |
| `stain_min_ref_images` | `6` | Minimum successful estimates needed to build a batch reference |

---

## DAB Signal Diagnostics

Before training, inspect DAB signal quality across the dataset:

```bash
python scripts/inspect_expression_signal.py \
  --input data/raw/train/IHC \
  --output-dir outputs/signal_debug \
  --max-figures 20
```

This saves a CSV with per-image IOD / mIOD values and debug figures showing the
RGB image, DAB heatmap, and tissue mask. Use it to tune `od_threshold`,
`min_iod_signal`, and `min_miod_signal`.

---

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- albumentations
- scikit-image
- scipy
- pandas
- matplotlib
- lpips
- pyyaml
- tqdm

See `requirements.txt` for pinned versions.
