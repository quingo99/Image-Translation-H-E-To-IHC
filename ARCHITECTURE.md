# Architecture & Design Document

## 1. Overview

The pipeline translates H&E (hematoxylin & eosin) whole-slide image patches into
synthetic HER2 IHC (immunohistochemistry) patches using a conditional GAN. Two
variants are trained:

- **M_base**: adversarial loss + Gaussian pyramid L1
- **M_expr**: M_base + clinically guided DAB expression loss

```
H&E patch (256×256)
       │
       ▼
  ┌──────────┐     ┌───────────────┐
  │ Generator│────►│ Generated IHC │──────────────────────┐
  │  (U-Net) │     └───────────────┘                      │
  └──────────┘             │                              │
       ▲                   │              ┌───────────────▼──────────────┐
       │              ┌────▼─────┐        │       Expression Loss        │
  real H&E            │Real IHC  │        │  Macenko stain separation    │
                      └────┬─────┘        │  IOD/mIOD relative error     │
                            │             └──────────────────────────────┘
                   ┌────────▼────────┐
                   │  Discriminator  │
                   │  (PatchGAN)     │
                   └─────────────────┘
```

**Total generator loss:**

```
L_G = λ_adv · L_adv + λ_pyr · L_pyr + λ_expr · L_expr
    = 5 · L_adv  +  25 · L_pyr  +  2.2 · L_expr
```

`L_expr` is added starting at `epoch_add_expr` (M_base always keeps it zero).

---

## 2. Generator — U-Net

### Architecture

A 6-level encoder–decoder U-Net with skip connections and a residual bottleneck.
All feature maps use instance normalization in the generator.

```
Input: (B, 3, 256, 256)  — H&E RGB in [-1, 1]

Encoder (stride-2 convolutions, halve spatial size each step):
  down1: (B,  64, 128, 128)   — no norm (raw pixels)
  down2: (B, 128,  64,  64)
  down3: (B, 256,  32,  32)
  down4: (B, 512,  16,  16)
  down5: (B, 512,   8,   8)
  down6: (B, 512,   4,   4)   — no norm (bottleneck is tiny)

Bottleneck (6 residual blocks at 4×4):
  b:    (B, 512,   4,   4)

Decoder (2× upsample + 3×3 conv, then concat skip):
  up1:  (B, 1024,  8,   8)  ← cat(512 from up, 512 from down5)
  up2:  (B, 1024, 16,  16)  ← cat(512,          512 from down4)  [dropout]
  up3:  (B,  768, 32,  32)  ← cat(512,          256 from down3)
  up4:  (B,  384, 64,  64)  ← cat(256,          128 from down2)
  up5:  (B,  192,128, 128)  ← cat(128,           64 from down1)

Final (2× upsample + 3×3 conv + Tanh):
  out:  (B,   3, 256, 256)  — IHC RGB in [-1, 1]
```

### Key design choices

- **Instance normalization** in the generator: better style transfer than batch norm, unaffected by batch statistics.
- **No norm on down1 and down6**: raw pixel inputs shouldn't be normalized; at 4×4 there is too little spatial information.
- **Dropout on up1, up2**: adds stochasticity in the early decoder stages (standard U-Net practice for image translation).
- **Reflective padding** on all convolutions: avoids border artifacts from zero-padding.
- **Residual bottleneck**: 6 residual blocks at the 4×4 spatial bottleneck allow the network to reason about global stain composition before decoding.

---

## 3. Discriminator — PatchGAN

A conditional PatchGAN that receives `[H&E, IHC]` concatenated along channels
and classifies 70×70 overlapping patches as real or fake.

```
Input: (B, 6, 256, 256)  — cat(H&E, IHC) along dim=1

  Conv(6→64,   stride=2, no norm) → LeakyReLU(0.2)     → (B,  64, 128, 128)
  Conv(64→128, stride=2)          → Norm → LeakyReLU   → (B, 128,  64,  64)
  Conv(128→256,stride=2)          → Norm → LeakyReLU   → (B, 256,  32,  32)
  Conv(256→512,stride=1)          → Norm → LeakyReLU   → (B, 512,  31,  31)
  Conv(512→1,  stride=1)                                → (B,   1,  30,  30)
```

Output is a 30×30 map of logits (one per patch). The MSE loss is applied
element-wise against a target filled with 0 (fake) or 1 (real).

**Spectral normalization** is applied to all convolutions (`spectral_norm_D: true`),
stabilizing training by constraining the Lipschitz constant of the discriminator.

**Discriminator is updated every 2 generator steps** (`d_update_every: 2`) and
its parameters are frozen (`requires_grad_(False)`) during the generator update
step to prevent unnecessary computation.

---

## 4. Loss Functions

### 4.1 Adversarial Loss — `L_adv`

MSE (LSGAN) loss. Compared to BCE, MSE gives smoother gradients and avoids
vanishing gradients for the generator.

```
L_D = 0.5 · MSE(D(x, y), 1)  +  0.5 · MSE(D(x, ŷ), 0)
L_adv = MSE(D(x, ŷ), 1)
```

### 4.2 Gaussian Pyramid L1 — `L_pyr`

A multi-scale L1 loss that computes pixel-level agreement at several resolutions.
This provides coarse-to-fine supervision and tolerance to small spatial
misalignments between patches.

**Gaussian kernel** (separable, 1D):
```
k = [1, 4, 6, 4, 1] / 16
```

**Pyramid construction** for `S` levels:
```
P_0 = image (original)
P_s = downsample(P_{s-1}) by blurring with k × k then stride-2 sample
```

**Loss:**
```
L_pyr = Σ_s  w_s · L1(P_s(y), P_s(ŷ))
where w_s = 1 / 2^s   (coarser scales weighted less)
```

With `pyramid_levels=3`, this operates at scales 1×, 0.5×, and 0.25×.

### 4.3 Expression Loss — `L_expr` (M_expr only)

Clinically guided loss that penalizes discrepancy in DAB (3,3'-diaminobenzidine)
staining between real and generated IHC. DAB encodes HER2 protein expression.

**Activation:** added to the total loss starting at `epoch_add_expr`.

**Pipeline per batch:**

```
1. Convert generated ŷ and real y from [-1,1] to [0,1]
2. Estimate tissue mask from real y (detached)           → mask (B, H, W)
3. Estimate Macenko stain matrix V from real y (detached) → V_i (3, 2) per image
4. Extract DAB map from real y using V_i    (detached)   → dab_y (H, W)
5. Extract DAB map from generated ŷ using V_i            → dab_ŷ (H, W)  [carries grad]
6. Compute global + spatial relative errors
```

**Global terms** (IOD = integrated optical density, mIOD = mean IOD per tissue area):
```
IOD(I) = Σ_p  mask_p · DAB_p(I)
mIOD(I) = IOD(I) / (Σ_p mask_p + ε)

rel_sym(a, b) = 2|a - b| / (|a| + |b| + ε)   ← symmetric relative error (SMAPE-style)

L_global = rel_sym(IOD(ŷ), IOD(y))  +  rel_sym(mIOD(ŷ), mIOD(y))
```

Images with very low signal (`IOD < min_iod_signal` or `mIOD < min_miod_signal`)
are skipped for the corresponding term to avoid outlier-dominated gradients.

**Spatial term** (8×8 patch grid):
```
Pool mask, dab_y, dab_ŷ to (8, 8) using adaptive average pooling.
For each patch c with sufficient tissue (pooled_mask >= 0.1) and signal:
    L_spatial_c = rel_sym(mIOD_c(ŷ), mIOD_c(y))

L_spatial = mean over active patches c
```

**Combined:**
```
L_expr = L_global + λ_spatial · L_spatial
       = L_global + 0.4 · L_spatial
```

All relative error terms are clamped to `max_rel_term=5.0` to prevent outliers
from dominating.

**Gradient design:**
- Tissue mask: computed from real `y` only, detached → no grad
- Stain matrix `V`: estimated from real `y` only, detached → no grad
- `dab_y` (real): detached → no grad
- `dab_ŷ` (generated): full gradient flows back through the DAB extraction and into the generator

---

## 5. Macenko Stain Separation

Standard Macenko (2009) method adapted for PyTorch with batch support and
gradient-safe reference stabilization.

### 5.1 RGB to Optical Density

```
I0 = quantile(I, 0.99)   per channel, per image   (≈ background white point)
I_norm = clamp(I / I0, min=ε, max=1.0)
OD = -log10(I_norm)
```

The `max=1.0` clamp enforces physically valid OD ≥ 0 (pixels brighter than the
background cannot have negative absorption).

`I0` is detached before the division to prevent gradients flowing through the
adaptive normalizer into a noisy quantile-selection gradient.

### 5.2 Stain Matrix Estimation (per image)

```
1. Extract tissue pixels: OD magnitude > od_threshold, discard top 0.5% outliers
2. Center the OD pixel cloud
3. SVD → take top-2 principal components (the OD distribution lives in a 2D plane)
4. Project each tissue pixel onto the 2D plane → angles θ
5. Stain vectors = extremes at 1st and 99th percentile angles (θ_1, θ_99)
6. Back-project to 3D OD space, L2-normalize
7. Canonicalize: assign columns to [H, DAB] order and resolve sign ambiguity
   by comparing cosine similarity with a fixed reference matrix (Ruifrok 2001)
```

Returns `None` if fewer than 10 tissue pixels are found or if the angular
spread `θ_99 − θ_1 < min_angle_spread`.

### 5.3 Batch Reference Stabilization

When enough images in a batch have valid estimates (`min_ref_images=6`):

```
V_ref = median (or mean) of valid per-image estimates
V_final = canonicalize((1 - blend) · V_img + blend · V_ref)
```

`stain_ref_blend=0.25` pulls each per-image estimate 25% toward the batch
consensus, reducing noise from low-stained or artifact-heavy images.

Images whose estimation fails fall back to `V_ref` if available, otherwise to
the fixed `DEFAULT_STAIN_MATRIX` (Ruifrok & Johnston 2001 H+DAB values).

### 5.4 DAB Extraction

```
OD = V @ S      (OD pixel = stain_matrix × stain_amounts)
S = pinv(V) @ OD_flat    (pseudoinverse unmixing, all pixels at once)
DAB = S[1]               (second column of V is DAB by convention)
DAB = softplus(DAB, β=8) (smooth nonnegativity, better gradients than hard clamp)
```

---

## 6. Data Pipeline

### Normalization

All images are stored as uint8 (0–255). The transform pipeline converts to
`[-1, 1]` via:
```
normalized = (pixel / 255.0 - 0.5) / 0.5
```

All model outputs (Tanh) and inputs stay in `[-1, 1]`. Stain and metric
operations convert to `[0, 1]` internally.

### Training augmentations

- Random crop to `image_size × image_size` (default 256×256)
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- Random 90° rotation (p=0.5)
- Same random transform applied identically to HE and IHC to preserve pairing

### Validation / Evaluation

- No augmentation
- Center crop to `val_image_size` during training validation (default 512×512)
- Full-resolution inference during `eval.py` (`eval_full_resolution: true`)

---

## 7. Metrics

### 7.1 Image Similarity

| Metric | Implementation | Direction |
|---|---|---|
| PSNR | `skimage.metrics.peak_signal_noise_ratio` on uint8 RGB | ↑ higher is better |
| SSIM | `skimage.metrics.structural_similarity` on uint8 RGB | ↑ |
| LPIPS | Pretrained AlexNet (`lpips` package), images in [-1,1] | ↓ lower is better |

### 7.2 DAB Expression

Measures how well the generated IHC preserves HER2 protein expression.

```
IOD error  = |IOD(ŷ) − IOD(y)|  / max(|IOD(y)|, 0.01)
mIOD error = |mIOD(ŷ) − mIOD(y)| / max(|mIOD(y)|, 0.001)
```

Stain matrix estimated from the real image; the same matrix is applied to both
real and generated to isolate the expression difference.

### 7.3 Structure Metrics

**Nuclei density error** — proxy for cell detection accuracy:
```
threshold = mean + std of hematoxylin concentration on tissue pixels (real image)
nuclei_frac(I) = fraction of tissue pixels with hema > threshold
error = |nuclei_frac(ŷ) − nuclei_frac(y)|
```

**Membrane intensity error** — proxy for membrane staining sharpness:
```
edge_energy(DAB) = mean Sobel gradient magnitude on tissue pixels
error = |edge_energy(DAB(ŷ)) − edge_energy(DAB(y))|
```

Both use the tissue mask derived from the real image to ensure fair comparison.

---

## 8. Training Strategy

| Aspect | Detail |
|---|---|
| Optimizer | Adam (β₁=0.5, β₂=0.999) for both G and D |
| LR schedule | Linear decay to 0 over all epochs |
| D / G update ratio | D updated every 2 G steps |
| Validation start | Epoch 45 (model has stabilized) |
| Validation frequency | Every 5 epochs |
| Expression loss activation | Epoch 20 (M_expr only) |
| Batch size | 8 (256×256 patches) |
| Total epochs | 100 |

### Best-checkpoint selection

`generator_best.pth` tracks the primary metric (`best_metric`):
- M_base: `"lpips"` (lower is better)
- M_expr: `"expr"` (expression score = 1 / avg_rel_error, higher is better)

Extra checkpoints track PSNR, LPIPS, and expression score independently.

---

## 9. File Relationships

```
train.py
  ├── src/data/bci_dataset.py        (data loading & augmentation)
  ├── src/models/pix2pix.py          (Generator, PatchDiscriminator)
  ├── src/losses/pyramid.py          (PyramidL1Loss)
  ├── src/losses/expression.py       (ExpressionLoss)
  │     └── src/stains/macenko.py    (rgb_to_od, estimate_stain_matrix, get_dab_map)
  ├── src/metrics/metrics.py         (compute_psnr, compute_dab_metrics, …)
  │     └── src/stains/macenko.py
  └── src/utils/repro.py             (seed_everything, save_run_metadata)

eval.py
  ├── src/data/bci_dataset.py
  ├── src/models/pix2pix.py          (Generator only)
  └── src/metrics/metrics.py

report.py
  └── (reads CSV files produced by eval.py, no model code)
```
