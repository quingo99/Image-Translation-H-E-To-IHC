"""Evaluation metrics for Phase 2.

Sections:
  9.1  Image similarity: PSNR, SSIM (RGB)
  9.2  Protein expression: IOD, mIOD on DAB maps
  9.3  Cell/membrane structure: nuclei density error, membrane intensity error
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import sobel

from src.stains.macenko import (
    rgb_to_od,
    estimate_stain_matrix,
    get_dab_map,
    tissue_mask_from_od,
    EPS,
)

MIN_IOD_DEN = 1e-2
MIN_MIOD_DEN = 1e-3


# 
#  Tensor helpers
# 

def tanh_to_uint8(t):
    """Convert (B, 3, H, W) tensor in [-1,1] to (B, H, W, 3) uint8 numpy."""
    arr = ((t + 1.0) / 2.0).clamp(0, 1).cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    return arr.transpose(0, 2, 3, 1)  # BCHW -> BHWC


def tanh_to_01(t):
    return (t + 1.0) / 2.0


# 
#  9.1  Image similarity
# 

def compute_psnr(y, y_hat):
    """PSNR on RGB, per-image then averaged."""
    y_np = tanh_to_uint8(y)
    yh_np = tanh_to_uint8(y_hat)
    vals = []
    for i in range(y_np.shape[0]):
        vals.append(peak_signal_noise_ratio(y_np[i], yh_np[i], data_range=255))
    return vals


def compute_ssim(y, y_hat):
    """SSIM on RGB, per-image then averaged."""
    y_np = tanh_to_uint8(y)
    yh_np = tanh_to_uint8(y_hat)
    vals = []
    for i in range(y_np.shape[0]):
        vals.append(structural_similarity(
            y_np[i], yh_np[i], data_range=255, channel_axis=2
        ))
    return vals


def load_lpips_model(device, net="alex"):
    """Load LPIPS model once for evaluation.

    Returns:
        LPIPS model on `device`, or None if `lpips` package is unavailable.
    """
    try:
        import lpips  # type: ignore
    except ImportError:
        return None

    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def compute_lpips(y, y_hat, lpips_model):
    """LPIPS distance on RGB in [-1, 1], per-image (lower is better)."""
    if lpips_model is None:
        return [float("nan")] * y.shape[0]

    with torch.no_grad():
        vals = lpips_model(y, y_hat)
    return vals.flatten().detach().cpu().tolist()


# 
#  9.2  Protein expression from DAB
# 

def compute_dab_metrics(
    y,
    y_hat,
    od_threshold=0.15,
    dilate_radius=2,
    dab_nonnegative="softplus",
    softplus_beta=10.0,
    stain_reference_mode="batch_avg",
    stain_ref_blend=0.0,
    stain_min_ref_images=6,
):
    """Compute per-image DAB expression metrics.

    Returns list of dicts with keys:
        iod_real, iod_gen, iod_abs_err, iod_rel_err,
        miod_real, miod_gen, miod_abs_err, miod_rel_err
    """
    # Convert from Tanh output range [-1,1] to [0,1] for stain operations
    y_01 = tanh_to_01(y).detach()
    yh_01 = tanh_to_01(y_hat).detach()
    B = y.shape[0]

    # Build tissue mask and stain matrix from REAL images only.
    # Using real images as reference ensures generated metrics are
    # measured on the same tissue region and with the same color basis.
    mask = tissue_mask_from_od(y_01, od_threshold, dilate_radius)  # (B, H, W)
    V_list = estimate_stain_matrix(
        y_01,
        od_threshold,
        reference_mode=stain_reference_mode,
        ref_blend=stain_ref_blend,
        min_ref_images=stain_min_ref_images,
    )

    results = []
    for i in range(B):
        V = V_list[i]
        # Extract DAB concentration maps via linear color deconvolution
        dab_y = get_dab_map(
            y_01[i],
            V,
            nonnegative=dab_nonnegative,
            softplus_beta=softplus_beta,
        ).cpu().numpy()
        dab_yh = get_dab_map(
            yh_01[i],
            V,
            nonnegative=dab_nonnegative,
            softplus_beta=softplus_beta,
        ).cpu().numpy()
        m = mask[i].cpu().numpy()

        # IOD: total DAB signal on tissue (sum of DAB * mask)
        # mIOD: mean DAB per tissue pixel (IOD / tissue area)
        tissue_area = m.sum() + 1e-8
        iod_y = (m * dab_y).sum()
        iod_yh = (m * dab_yh).sum()
        miod_y = iod_y / tissue_area
        miod_yh = iod_yh / tissue_area
        # Symmetric relative error (SMAPE-style): 2|a-b| / (|a| + |b|)
        # Matches the formula used in ExpressionLoss so training and evaluation
        # measure the same quantity.
        iod_den = max(abs(iod_y) + abs(iod_yh), MIN_IOD_DEN)
        miod_den = max(abs(miod_y) + abs(miod_yh), MIN_MIOD_DEN)

        results.append({
            "iod_real": float(iod_y),
            "iod_gen": float(iod_yh),
            "iod_abs_err": float(abs(iod_yh - iod_y)),
            "iod_rel_err": float(2.0 * abs(iod_yh - iod_y) / iod_den),
            "miod_real": float(miod_y),
            "miod_gen": float(miod_yh),
            "miod_abs_err": float(abs(miod_yh - miod_y)),
            "miod_rel_err": float(2.0 * abs(miod_yh - miod_y) / miod_den),
        })

    return results


# 
#  9.3  Cell/membrane structure metrics
# 

def _hematoxylin_area_fraction(img_01, V, mask, threshold=None):
    """Nuclei proxy: fraction of tissue area with high hematoxylin concentration.

    Nuclei stain strongly with hematoxylin (H channel after color deconvolution).
    Pixels above mean+1std of tissue hematoxylin values are treated as nuclei.

    Args:
        img_01: (3, H, W) image in [0, 1].
        V: (3, 2) stain matrix.
        mask: (H, W) tissue mask.
        threshold: if provided, use this threshold instead of computing from img_01.
                   This ensures real and generated images are compared with the same threshold.

    Returns:
        area_fraction: float, fraction of tissue with high hematoxylin.
        threshold: float, the threshold used (for passing to the generated image call).
    """
    # Deconvolve to get per-pixel stain amounts: S[0]=hematoxylin, S[1]=DAB
    od = rgb_to_od(img_01)
    od_flat = od.reshape(3, -1)
    V_pinv = torch.linalg.pinv(V)
    S = V_pinv @ od_flat  # (2, HW)
    H, W = img_01.shape[1], img_01.shape[2]
    hema = S[0].reshape(H, W).clamp(min=0)  # hematoxylin channel (non-negative)

    # Adaptive threshold: mean + 1 std on tissue pixels (computed from real image only).
    # The same threshold is reused for the generated image so comparison is fair.
    tissue_vals = hema[mask > 0.5]
    if tissue_vals.numel() < 10:
        return 0.0, 0.0
    if threshold is None:
        threshold = float((tissue_vals.mean() + tissue_vals.std()).item())
    nuclei_mask = (hema > threshold) & (mask > 0.5)
    return float(nuclei_mask.float().sum() / (mask.sum() + 1e-8)), threshold


def compute_nuclei_density_error(
    y,
    y_hat,
    od_threshold=0.15,
    dilate_radius=3,
    stain_reference_mode="batch_avg",
    stain_ref_blend=0.0,
    stain_min_ref_images=6,
):
    """Nuclei density error: |area_frac(y) - area_frac(y_hat)|."""
    y_01 = tanh_to_01(y).detach()
    yh_01 = tanh_to_01(y_hat).detach()
    B = y.shape[0]
    mask = tissue_mask_from_od(y_01, od_threshold, dilate_radius)
    V_list = estimate_stain_matrix(
        y_01,
        od_threshold,
        reference_mode=stain_reference_mode,
        ref_blend=stain_ref_blend,
        min_ref_images=stain_min_ref_images,
    )

    errors = []
    for i in range(B):
        V = V_list[i]
        frac_y, thresh = _hematoxylin_area_fraction(y_01[i], V, mask[i])
        frac_yh, _ = _hematoxylin_area_fraction(yh_01[i], V, mask[i], threshold=thresh)
        errors.append(abs(frac_y - frac_yh))
    return errors


def _dab_edge_energy(dab_np, mask_np):
    """Membrane proxy: mean Sobel edge energy of DAB on tissue pixels.

    HER2-positive membranes produce sharp DAB boundaries. High edge energy
    in the DAB map correlates with strong membrane staining patterns.
    """
    # Sobel filters approximate image gradients in x and y directions
    sx = sobel(dab_np, axis=0)
    sy = sobel(dab_np, axis=1)
    # Gradient magnitude = edge "sharpness" at each pixel
    edge = np.sqrt(sx ** 2 + sy ** 2)
    tissue_idx = mask_np > 0.5
    if tissue_idx.sum() < 10:
        return 0.0
    # Average edge energy over tissue pixels only (ignore background)
    return float(edge[tissue_idx].mean())


def compute_membrane_intensity_error(
    y,
    y_hat,
    od_threshold=0.15,
    dilate_radius=3,
    dab_nonnegative="softplus",
    softplus_beta=10.0,
    stain_reference_mode="batch_avg",
    stain_ref_blend=0.0,
    stain_min_ref_images=6,
):
    """Membrane intensity error: |edge_energy(DAB_y) - edge_energy(DAB_yhat)|."""
    y_01 = tanh_to_01(y).detach()
    yh_01 = tanh_to_01(y_hat).detach()
    B = y.shape[0]
    mask = tissue_mask_from_od(y_01, od_threshold, dilate_radius)
    V_list = estimate_stain_matrix(
        y_01,
        od_threshold,
        reference_mode=stain_reference_mode,
        ref_blend=stain_ref_blend,
        min_ref_images=stain_min_ref_images,
    )

    errors = []
    for i in range(B):
        V = V_list[i]
        dab_y = get_dab_map(
            y_01[i],
            V,
            nonnegative=dab_nonnegative,
            softplus_beta=softplus_beta,
        ).cpu().numpy()
        dab_yh = get_dab_map(
            yh_01[i],
            V,
            nonnegative=dab_nonnegative,
            softplus_beta=softplus_beta,
        ).cpu().numpy()
        m = mask[i].cpu().numpy()

        ee_y = _dab_edge_energy(dab_y, m)
        ee_yh = _dab_edge_energy(dab_yh, m)
        errors.append(abs(ee_y - ee_yh))
    return errors
