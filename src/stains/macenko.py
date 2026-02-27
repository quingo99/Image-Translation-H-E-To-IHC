"""Macenko-style stain separation utilities for H + DAB images.

This module estimates per-image stain bases in optical density space and
extracts DAB concentration maps used by expression losses and metrics.

Also, the deconvolution method is also referred by paper "Multi-target stain normalization for histology slides"
"""

import torch
import torch.nn.functional as F


EPS = 1e-6

# Default stain vectors for H & DAB (normalized OD), from Ruifrok & Johnston (2001);
# matches the ImageJ/Fiji Colour Deconvolution "H DAB" preset.
DEFAULT_STAIN_MATRIX = torch.tensor(
    [
        [0.6500, 0.2700],
        [0.7040, 0.5700],
        [0.2860, 0.7760],
    ],
    dtype=torch.float32,
)


def rgb_to_od(img):
    """Convert RGB intensities to optical density.

    Args:
        img: (3, H, W) or (B, 3, H, W) in [0, 1].

    Returns:
        OD image with the same shape as input.
    """
    if img.ndim == 4:
        # Batched: (B, 3, H, W)
        flat = img.reshape(img.shape[0], 3, -1)  # (B, 3, HW)
        I0 = torch.quantile(flat, 0.99, dim=2).clamp(min=EPS)  # (B, 3)
        img_norm = (img / I0[..., None, None]).clamp(min=EPS, max=1.0)
    elif img.ndim == 3:
        # Single image: (3, H, W)
        flat = img.reshape(3, -1)
        I0 = torch.quantile(flat, 0.99, dim=1).clamp(min=EPS)  # (3,)
        img_norm = (img / I0[:, None, None]).clamp(min=EPS, max=1.0)
    else:
        raise ValueError(f"Expected 3D or 4D input, got {img.ndim}D")
    return -torch.log10(img_norm)


def _normalize_columns(V):
    return V / (V.norm(dim=0, keepdim=True) + EPS)


def _canonicalize_stain_matrix(V, V_ref=None):
    """
    Make an estimated 3x2 stain basis deterministic and consistent.

    Why needed:
      - SVD/PCA-based estimation has (1) sign ambiguity: v and -v are equivalent,
        and (2) ordering ambiguity: the two stain columns can come out swapped.
      - Downstream code assumes columns are [H, DAB], so we force that convention.

    Strategy:
      - Compare the two estimated columns to a reference [H, DAB] basis using cosine similarity.
      - Choose the best assignment (keep vs swap).
      - Flip signs to align each column with the reference direction.
    """
    # Normalize columns so dot products behave like cosine similarity.
    Vn = _normalize_columns(V)

    # Pick the reference basis used to define "column 0 is H" and "column 1 is DAB".
    if V_ref is None:
        V_ref = DEFAULT_STAIN_MATRIX.to(device=V.device, dtype=V.dtype)
    Vrn = _normalize_columns(V_ref)

    # 2x2 similarity matrix:
    #   sim[i,j] = |cosine_similarity( Vn[:, i], Vrn[:, j] )|
    # The abs() removes sign ambiguity (v vs -v) for the purpose of assignment.
    sim = torch.abs(Vn.T @ Vrn)  # (2,2)

    # Two possible assignments for 2 stains:
    #   keep: col0->H and col1->DAB  score = sim[0,0] + sim[1,1]
    #   swap: col0->DAB and col1->H  score = sim[0,1] + sim[1,0]
    keep_score = sim[0, 0] + sim[1, 1]
    swap_score = sim[0, 1] + sim[1, 0]
    if swap_score > keep_score:
        Vn = Vn[:, [1, 0]]  # swap columns so they best match [H, DAB]

    # Now that the order matches the reference, fix the sign for each column:
    # If a column points opposite the reference (dot < 0), flip it.
    for j in range(2):
        if torch.dot(Vn[:, j], Vrn[:, j]) < 0:
            Vn[:, j] = -Vn[:, j]

    # Re-normalize (usually unnecessary but harmless) and return.
    return _normalize_columns(Vn)




def _default_stain_matrix(device, dtype):
    return DEFAULT_STAIN_MATRIX.to(device=device, dtype=dtype)


def _estimate_stain_matrix_single(
    od_pixels,
    min_pixels=200,
    min_angle_spread=0.20,
):
    """
    Estimate a 3x2 stain basis (OD-RGB) from a set of tissue pixels using a Macenko-style method.

    Idea (Macenko-style):
      - In optical density (OD) space, stain mixing is approximately linear.
      - For two stains (H and DAB), tissue OD vectors roughly lie near a 2D subspace (a plane) in R^3.
      - We find that plane via SVD/PCA on centered OD pixels.
      - We project pixels onto the plane, compute their angles, and take angular extremes (robust percentiles)
        as the two stain directions.
      - Finally, we canonicalize the result to a consistent [H, DAB] order and sign.

    Args:
        od_pixels: (N,3) OD vectors sampled from tissue pixels.
        min_pixels: minimum number of pixels needed for a stable estimate.
        min_angle_spread: minimum angular spread (radians) required; prevents degenerate estimates.

    Returns:
        V: (3,2) stain matrix with columns [H, DAB], or None if estimation is unreliable.
    """
    # 1) Bail out if not enough pixels: SVD + quantiles become unstable with tiny N.
    if od_pixels.shape[0] < min_pixels:
        return None

    # 2) Center the data (PCA/SVD is usually done on mean-centered samples).
    centered = od_pixels - od_pixels.mean(dim=0, keepdim=True)

    # 3) SVD: right singular vectors (Vh) are principal directions in OD space.
    #    The first two principal components span the best-fit 2D plane for the data.
    #    Upcast to float32 for numerical stability (SVD on float16 can be unreliable).
    svd_input = centered.float() if centered.dtype != torch.float32 else centered
    try:
        _, _, Vh = torch.linalg.svd(svd_input, full_matrices=False)
    except Exception:
        # If SVD fails (rare, but can happen on some dtypes/devices), fall back.
        return None
    # Cast back to original dtype
    Vh = Vh.to(dtype=centered.dtype)

    # 4) "plane" is a 2x3 matrix whose rows are the top-2 principal directions.
    #    Project each centered pixel onto this 2D coordinate system.
    plane = Vh[:2, :]            # (2,3)
    proj = centered @ plane.T    # (N,2) coordinates in the PCA plane

    # 5) Convert 2D coordinates to angles in the plane.
    #    Each pixel gets an angle; stain directions tend to occur at angular extremes.
    angles = torch.atan2(proj[:, 1], proj[:, 0])  # in [-pi, pi]

    # 6) Angles are circular. If the distribution straddles -pi/+pi, naive percentiles break.
    #    Unwrap angles around a reference (median) so percentiles behave correctly.
    ref = torch.median(angles)
    two_pi = 2 * torch.pi
    angles_u = torch.remainder(angles - ref + torch.pi, two_pi) - torch.pi  # centered to [-pi, pi)

    # 7) Take robust angular extremes via percentiles (ignore outliers).
    #    These correspond to two stain directions in the PCA plane.
    p01 = torch.quantile(angles_u, 0.01) + ref
    p99 = torch.quantile(angles_u, 0.99) + ref

    # 8) If the angle spread is too small, the stains are not separable (degenerate case).
    if (p99 - p01) < min_angle_spread:
        return None

    # 9) Convert extreme angles back to unit direction vectors in the 2D plane coordinates.
    dir1 = torch.stack([torch.cos(p01), torch.sin(p01)])  # (2,)
    dir2 = torch.stack([torch.cos(p99), torch.sin(p99)])  # (2,)

    # 10) Map these 2D directions back into 3D OD space (plane.T is 3x2).
    v1 = plane.T @ dir1  # (3,)
    v2 = plane.T @ dir2  # (3,)

    # 11) Normalize the resulting OD direction vectors.
    v1 = v1 / (v1.norm() + EPS)
    v2 = v2 / (v2.norm() + EPS)

    # 12) Stack into a 3x2 matrix and canonicalize:
    #     - ensures consistent column order [H, DAB]
    #     - fixes sign ambiguity (v and -v are equivalent in PCA)
    V = torch.stack([v1, v2], dim=1)  # (3,2)
    return _canonicalize_stain_matrix(V)

def _collect_tissue_od_pixels(img_01, od_threshold):
    """
    Extract a set of OD-space pixels likely belonging to tissue (not background).

    Steps:
      1) Convert RGB->[0,1] to optical density (OD). Background/white has OD ~ 0.
      2) Flatten all pixels to an (HW, 3) list of OD vectors.
      3) Compute OD magnitude per pixel: ||OD||. Tissue tends to have larger magnitude.
      4) Keep pixels with ||OD|| > od_threshold (remove bright background).
      5) Additionally remove extreme-high OD outliers (e.g., dust, pen marks, saturation)
         by dropping the top 0.5% of tissue magnitudes.

    Args:
        img_01: (3, H, W) RGB image in [0, 1].
        od_threshold: scalar threshold on OD magnitude used to separate tissue from background.

    Returns:
        od_tissue: (N, 3) OD pixel vectors (N can vary by image).
    """
    # OD = -log10(I). Shape: (3,H,W)
    od = rgb_to_od(img_01)

    # Flatten to per-pixel OD vectors: (HW,3)
    od_flat = od.permute(1, 2, 0).reshape(-1, 3)

    # Per-pixel OD magnitude
    od_mag = od_flat.norm(dim=1)

    # Tissue selection: larger OD magnitude than background
    tissue_mask = od_mag > od_threshold

    # Remove a small fraction of very high-OD outliers (robustness)
    if tissue_mask.any():
        hi = torch.quantile(od_mag[tissue_mask], 0.995)
        tissue_mask = tissue_mask & (od_mag <= hi)

    return od_flat[tissue_mask]


def _estimate_reference_from_batch(estimated_list, min_ref_images, reference_mode):
    """
    Build a batch-level reference stain matrix from valid per-image estimates.

    Motivation:
      - Per-image Macenko estimates can be noisy or fail for some images.
      - If enough images in the batch yield valid estimates, aggregating them provides
        a more stable reference basis (V_ref) for optional blending or fallback.

    Assumptions:
      - Each V in estimated_list is already canonicalized to the same convention
        (columns [H, DAB] with consistent signs). This prevents cancellation when averaging.

    Args:
        estimated_list: list of (3,2) stain matrices or None (failed estimates).
        min_ref_images: minimum number of non-None estimates required to form V_ref.
        reference_mode: "batch_avg" or "batch_median".

    Returns:
        V_ref: (3,2) canonicalized reference stain matrix, or None if insufficient valid images.
    """
    # Keep only successful estimates
    valid = [V for V in estimated_list if V is not None]
    if len(valid) < int(min_ref_images):
        return None

    stacked = torch.stack(valid, dim=0)
    if reference_mode == "batch_avg":
        # Average across images (elementwise mean of 3x2 matrices)
        V_ref = stacked.mean(dim=0)
    elif reference_mode == "batch_median":
        # element-wise median across images (more robust than mean)
        V_ref = stacked.median(dim=0).values  # (3, 2)
    else:
        raise ValueError(
            "reference_mode must be one of {'batch_avg', 'batch_median'}."
        )

    # Canonicalize again (fix any small drift; ensures [H, DAB] convention)
    return _canonicalize_stain_matrix(V_ref)


def estimate_stain_matrix(
    y_01,
    od_threshold=0.15,
    reference_mode="batch_avg",
    ref_matrix=None,
    ref_blend=0.0,
    min_ref_images=6,
):
    """
    Estimate a 3x2 stain matrix V for each image in a batch.

    What this function does (high level):
      1) For each image, collect "tissue" pixels in optical density (OD) space.
      2) Run a Macenko-style estimator to get a per-image stain basis V_img (3x2).
         - Columns are canonicalized to be [H, DAB].
         - If estimation fails (too little tissue / poor angular spread), returns None.
      3) Optionally build or use a reference stain basis V_ref:
         - "none": no reference (pure per-image or fallback).
         - "batch_avg": average valid per-image estimates in this batch -> V_ref.
         - "batch_median": median valid per-image estimates in this batch -> V_ref.
         - "provided": user passes an explicit ref_matrix -> V_ref.
      4) For each image:
         - If estimation failed -> use V_ref if available else DEFAULT fallback.
         - If estimation succeeded and V_ref exists and ref_blend>0 -> blend V_img with V_ref.

    Args:
        y_01: (B, 3, H, W) RGB images in [0, 1].
        od_threshold: threshold on OD magnitude used to decide which pixels are "tissue".
        reference_mode:
            "none"      -> per-image Macenko; fallback to DEFAULT when needed.
            "batch_avg" -> compute V_ref by averaging valid per-image estimates in batch.
            "batch_median" -> compute V_ref by median over valid per-image estimates.
            "provided"  -> use the given ref_matrix as V_ref.
        ref_matrix: optional (3,2) reference stain matrix used when reference_mode="provided".
        ref_blend: float in [0,1]. If >0 and V_ref exists:
            V = (1-ref_blend)*V_img + ref_blend*V_ref  (then re-canonicalized).
            - 0.0 => use per-image only
            - 1.0 => force reference only (for images with valid estimate too)
        min_ref_images: minimum number of valid per-image estimates required to form V_ref
            in "batch_avg" and "batch_median" modes.

    Returns:
        V_out: list of length B, each element is a (3,2) stain matrix [H, DAB].
    """
    # Validate options
    if reference_mode not in {"none", "batch_avg", "batch_median", "provided"}:
        raise ValueError(
            "reference_mode must be one of {'none', 'batch_avg', 'batch_median', 'provided'}"
        )
    if not (0.0 <= float(ref_blend) <= 1.0):
        raise ValueError("ref_blend must be in [0, 1].")

    B = y_01.shape[0]
    device = y_01.device
    dtype = y_01.dtype

    # 1) Per-image stain estimation (may fail -> None)
    estimated = []
    for i in range(B):
        # Collect tissue OD pixels (N,3) by OD magnitude thresholding and outlier removal
        od_tissue = _collect_tissue_od_pixels(y_01[i], od_threshold)

        # Estimate 3x2 stain matrix from OD pixels using SVD + angular extremes
        # Returns None if insufficient pixels / unstable estimate
        estimated.append(_estimate_stain_matrix_single(od_tissue))

    # 2) Optional reference basis V_ref
    V_ref = None
    if reference_mode == "provided":
        if ref_matrix is None:
            raise ValueError("reference_mode='provided' requires ref_matrix.")
        # Canonicalize reference to ensure [H, DAB] and consistent signs
        V_ref = _canonicalize_stain_matrix(ref_matrix.to(device=device, dtype=dtype))

    elif reference_mode in {"batch_avg", "batch_median"}:
        # Build a robust reference from this batch if enough images estimated successfully
        V_ref = _estimate_reference_from_batch(estimated, min_ref_images, reference_mode)



    # 3) Produce final per-image matrices with fallback/blending
    V_out = []
    fallback = _default_stain_matrix(device, dtype)  # DEFAULT_STAIN_MATRIX on correct device/dtype

    for i, V_img in enumerate(estimated):
        if V_img is None:
            # If estimation failed: prefer reference (if available), else use default fallback
            V = V_ref if V_ref is not None else fallback
        else:
            V = V_img
            # If we have a reference and want some stabilization, blend and re-canonicalize
            if V_ref is not None and ref_blend > 0.0:
                V = _canonicalize_stain_matrix((1.0 - ref_blend) * V + ref_blend * V_ref)

        V_out.append(V)

    return V_out


def get_dab_map(img_01, V, nonnegative="softplus", softplus_beta=10.0):
    """
    Compute the per-pixel DAB "concentration" map via linear color deconvolution.

    Args:
        img_01: (3, H, W) RGB image in [0, 1].
        V:      (3, 2) stain matrix in OD-RGB space, columns are [H, DAB] unit vectors.
        nonnegative:
            "clamp"    -> enforce nonnegativity with hard ReLU: max(dab, 0).
            "softplus" -> enforce nonnegativity smoothly (better gradients).
            "none"     -> allow negative values (pure linear unmixing).
        softplus_beta: controls how close softplus is to a hard clamp (larger = sharper).

    Returns:
        dab_map: (H, W) DAB concentration map (one scalar per pixel).
    """
    # Spatial size (img_01 is channel-first: (3,H,W))
    H, W = img_01.shape[1], img_01.shape[2]

    # Convert RGB intensities to optical density (OD) where mixing is ~linear:
    # OD = -log10(I).  Darker stain -> larger OD.
    od = rgb_to_od(img_01)  # (3, H, W)

    # Flatten pixels so we can apply one matrix multiply to all pixels at once.
    # Each column is a 3-vector OD for one pixel: (3, HW)
    od_flat = od.reshape(3, -1)

    # In OD space, we assume: od_pixel ≈ V @ s_pixel
    # where s_pixel is a 2-vector of stain "amounts" [H_amount, DAB_amount].
    # Solve least-squares for s_pixel using the pseudoinverse:
    # s_pixel ≈ pinv(V) @ od_pixel
    V_pinv = torch.linalg.pinv(V)  # (2, 3)
    S = V_pinv @ od_flat           # (2, HW) stain amounts for all pixels

    # DAB is the 2nd stain by convention (index 1). Reshape back to image.
    dab = S[1].reshape(H, W)

    # Enforce nonnegativity (physical concentrations should be >= 0).
    if nonnegative == "clamp":
        # Hard constraint: negatives become 0 (but gradients are 0 for negative region).
        return dab.clamp(min=0)
    if nonnegative == "softplus":
        # Smooth constraint: always positive and keeps nonzero gradients near/below 0.
        return F.softplus(dab, beta=softplus_beta)
    if nonnegative == "none":
        # Raw linear unmixing result (may include negatives due to noise/model mismatch).
        return dab

    raise ValueError(f"Unsupported nonnegative mode: {nonnegative}")


def tissue_mask_from_od(y_01, od_threshold=0.15, dilate_radius=3):
    """
    Build a (rough) binary tissue mask from an RGB image batch using OD magnitude.

    Intuition:
      - Convert RGB in [0,1] to optical density (OD): OD = -log10(I).
      - Background/white regions have I ~ 1 -> OD ~ 0 (small magnitude).
      - Tissue/stained regions have lower intensities -> larger OD magnitude.
      - Thresholding OD magnitude separates tissue from background.
      - Optional dilation fills small holes and connects nearby tissue regions.

    Args:
        y_01: (B, 3, H, W) RGB images in [0, 1].
        od_threshold: scalar threshold on ||OD||. Higher -> stricter tissue selection.
        dilate_radius: radius (in pixels) for binary dilation. 0 disables dilation.

    Returns:
        mask: (B, H, W) float tensor in {0.0, 1.0} indicating tissue pixels.
    """
    # Convert RGB intensities to optical density. Shape: (B,3,H,W)
    od = rgb_to_od(y_01)

    # Per-pixel OD magnitude: sqrt(OD_R^2 + OD_G^2 + OD_B^2)
    # Background tends to have small magnitude; tissue tends to have larger magnitude.
    od_mag = od.norm(dim=1)  # (B, H, W)

    # Initial binary tissue mask via thresholding OD magnitude.
    mask = (od_mag > od_threshold).float()  # (B, H, W)

    # Optional dilation:
    # We implement dilation with a convolution over a square (2r+1)x(2r+1) ones kernel.
    # If ANY pixel in the neighborhood is 1, the output becomes 1.
    if dilate_radius > 0:
        r = int(dilate_radius)

        # Ones kernel acts like a neighborhood "OR" after thresholding (>0).
        # Match dtype of the mask to avoid dtype mismatch in conv2d.
        kernel = torch.ones(1, 1, 2 * r + 1, 2 * r + 1, device=y_01.device, dtype=mask.dtype)

        # conv2d expects (N, C, H, W), so add a channel dimension.
        mask_4d = mask.unsqueeze(1)  # (B, 1, H, W)

        # Convolution counts how many 1s are in each neighborhood. If count>0 => dilated mask=1.
        mask = (F.conv2d(mask_4d, kernel, padding=r) > 0).squeeze(1).float()  # (B,H,W)

    return mask