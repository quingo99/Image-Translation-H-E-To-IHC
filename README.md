# Phase 2: Pyramid Pix2Pix with Clinically Guided Expression Loss

Virtual HER2 IHC staining from H&E using a conditional GAN with multi-scale Gaussian pyramid supervision and optional DAB expression loss via Macenko stain separation.

## Project Structure

```text
project/
  configs/
    base.yaml          # M_base: Pyramid Pix2Pix
    expr.yaml          # M_expr: + expression loss L_expr
  src/
    data/bci_dataset.py       # Dataset for paired HE/IHC
    models/pix2pix.py         # Generator (U-Net) + PatchGAN discriminator
    losses/pyramid.py         # Gaussian pyramid L1 loss
    losses/expression.py      # Clinically guided expression loss
    stains/macenko.py         # Macenko stain separation + tissue masking
    metrics/metrics.py        # PSNR, SSIM, DAB, and structure metrics
    utils/repro.py            # Seeding, run metadata, run folders
  train.py                    # Training script
  eval.py                     # Evaluation script
  report.py                   # Report generator (tables/plots/hypothesis checks)
  run.py                      # Single entry point
  outputs/
    base/run_001/...
    expr/run_001/...
    report/
  data/raw/
    train/HE + IHC
    val/HE + IHC
    test/HE (+ optional IHC)
    groundtruth/              # Optional fallback IHC for test split
  requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Place data under `data/raw/` with paired filenames.

## How to Run

### Full pipeline

```bash
python run.py all
```

This will:
- train `M_base` and `M_expr`
- evaluate both models
- auto-generate the report

### Individual steps

```bash
# Train baseline
python run.py train --config configs/base.yaml

# Train expression model
python run.py train --config configs/expr.yaml

# Evaluate
python run.py eval --config configs/base.yaml --checkpoint outputs/base/run_001/generator_best.pth
python run.py eval --config configs/expr.yaml --checkpoint outputs/expr/run_005/generator_best.pth
```

### Evaluation outputs

`eval.py` tries `test` first, and falls back to `val` if paired test IHC is unavailable.

It writes:
- `metrics_test.csv` when evaluating test split
- `metrics_val.csv` when falling back to val split

under the run directory, for example:
- `outputs/base/run_001/metrics_test.csv`
- `outputs/expr/run_001/metrics_val.csv`

### Crop and resolution controls

Use these config keys to control patch sampling and full-resolution validation/evaluation:

- `data.image_size`: patch size used by train random crops (and by val/eval center crop when full-resolution mode is off).
- `data.train_crops_per_image`: number of random crops drawn per source train image per epoch.
- `data.train_crop_mode`: train crop strategy (`random` or `quadrant_jitter`; the latter enforces 2x2 quadrant coverage with jitter when sampling).
- `data.val_full_resolution`: when `true`, validation uses original resolution (no center crop).
- `data.eval_full_resolution`: when `true`, `eval.py` uses original resolution (no center crop).
- `training.val_batch_size`: validation batch size override (recommended `1` for full-resolution runs).
- `training.eval_batch_size`: eval batch size override (recommended `1` for full-resolution runs).
- `training.val_tiling_mode`: validation tiling mode (`none` or `2x2`).
- `training.val_tile_size`: tile size used when `training.val_tiling_mode: "2x2"` (e.g., `512` for 2x2 tiles on `1024x1024` images).
- `training.val_schedule`: validation trigger mode (`fixed` or `on_train_loss`).
- `training.val_start_epoch`: first epoch eligible for validation.
- `training.val_cooldown_epochs`: cooldown (in epochs) between validation runs for `on_train_loss`.
- `training.val_loss_min_delta`: minimum `loss_G` improvement required to trigger validation in `on_train_loss`.

### Generate report

Use the actual metrics files produced by each run:

```bash
python run.py report --base outputs/base/run_001/metrics_test.csv --expr outputs/expr/run_001/metrics_test.csv
```

If one model used val fallback, pass `metrics_val.csv` for that model.

## Models

- `M_base`: conditional GAN (U-Net generator + PatchGAN discriminator) with multi-scale Gaussian pyramid L1 loss.
- `M_expr`: `M_base` plus clinically guided expression loss (`L_expr`) activated at epoch 40.

## Metrics

| Category | Metrics |
|---|---|
| Image similarity | PSNR, SSIM, LPIPS (lower is better) |
| DAB expression | IOD error, mIOD error, Pearson-r |
| Cell structure | Nuclei density error, membrane intensity error |

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full list
