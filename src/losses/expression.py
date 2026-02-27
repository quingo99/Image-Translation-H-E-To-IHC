"""Expression loss: global intensity + patch-wise spatial DAB agreement.

L_expr = L_global + lambda_spatial * L_spatial

L_global = rel_sym(IOD(y_hat), IOD(y)) + rel_sym(mIOD(y_hat), mIOD(y))
L_spatial = mean_c rel_sym(mIOD_c(y_hat), mIOD_c(y)) over active spatial cells c

where:
    rel_sym(a, b) = 2|a - b| / (|a| + |b| + eps)
    IOD(I) = sum_p m_p * D_p(I)
    mIOD(I) = sum_p m_p * D_p(I) / (sum_p m_p + eps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.stains.macenko import estimate_stain_matrix, get_dab_map, tissue_mask_from_od

EPS = 1e-6
MIN_IOD_DEN = 1e-2
MIN_MIOD_DEN = 1e-3
MIN_SPATIAL_DEN = 1e-3
DEFAULT_MAX_REL_TERM = 10.0


def _compute_iod_miod(dab_map, mask):
    """Compute IOD and mIOD for a single image.

    Args:
        dab_map: (H, W) DAB concentration.
        mask: (H, W) tissue mask (float, 0/1).

    Returns:
        iod: scalar total DAB on tissue.
        miod: scalar mean DAB on tissue.
    """
    masked_dab = mask * dab_map
    tissue_area = mask.sum() + EPS
    iod = masked_dab.sum()
    miod = iod / tissue_area
    return iod, miod


def _symmetric_relative_error(pred, target, min_den):
    """Symmetric normalized error (SMAPE-style), bounded and stable near zero."""
    den = torch.clamp(torch.abs(pred) + torch.abs(target), min=min_den)
    return 2.0 * torch.abs(pred - target) / den


def _compute_patchwise_relative_miod(
    dab_y,
    dab_yhat,
    mask,
    grid_size,
    min_tissue_frac,
    min_signal,
    max_rel_term,
):
    """Patch-wise relative error between real and generated DAB mIOD maps.

    Pools mask, real DAB, and generated DAB in a single batched call for efficiency.
    """
    # Stack into a single 3-channel tensor and pool once instead of 3 separate calls.
    stacked = torch.stack([mask, dab_y * mask, dab_yhat * mask], dim=0)  # (3, H, W)
    pooled = F.adaptive_avg_pool2d(
        stacked.unsqueeze(0), (grid_size, grid_size)
    ).squeeze(0)  # (3, grid, grid)

    pooled_mask = pooled[0]
    pooled_real = pooled[1]
    pooled_gen = pooled[2]

    # Convert pooled masked sums to pooled tissue means (local mIOD).
    patch_miod_real = pooled_real / (pooled_mask + EPS)
    patch_miod_gen = pooled_gen / (pooled_mask + EPS)

    active = pooled_mask >= min_tissue_frac
    if min_signal > 0.0:
        active = active & (patch_miod_real.detach() >= min_signal)

    n_active = int(active.sum().item())
    if n_active == 0:
        return torch.tensor(0.0, device=dab_yhat.device), 0.0

    rel = _symmetric_relative_error(
        patch_miod_gen,
        patch_miod_real.detach(),
        min_den=max(min_signal, MIN_SPATIAL_DEN),
    )
    rel = torch.clamp(rel, max=max_rel_term)

    return rel[active].mean(), n_active / active.numel()


class ExpressionLoss(nn.Module):
    """Clinically guided expression loss using Macenko stain separation.

    Combines global IOD/mIOD relative error with optional patch-wise spatial
    agreement on DAB mIOD.
    """

    def __init__(
        self,
        od_threshold=0.15,
        dilate_radius=3,
        min_iod_signal=0.0,
        min_miod_signal=0.0,
        max_rel_term=DEFAULT_MAX_REL_TERM,
        dab_nonnegative="softplus",
        softplus_beta=10.0,
        stain_reference_mode="none",
        stain_ref_blend=0.0,
        stain_min_ref_images=4,
        spatial_weight=0.5,
        spatial_grid_size=16,
        spatial_min_tissue_frac=0.10,
        spatial_min_signal=0.0,
    ):
        super().__init__()
        self.od_threshold = od_threshold
        self.dilate_radius = dilate_radius
        self.min_iod_signal = float(min_iod_signal)
        self.min_miod_signal = float(min_miod_signal)
        self.max_rel_term = float(max_rel_term)
        self.dab_nonnegative = str(dab_nonnegative)
        self.softplus_beta = float(softplus_beta)
        self.stain_reference_mode = str(stain_reference_mode)
        self.stain_ref_blend = float(stain_ref_blend)
        self.stain_min_ref_images = int(stain_min_ref_images)
        self.spatial_weight = float(spatial_weight)
        self.spatial_grid_size = int(spatial_grid_size)
        self.spatial_min_tissue_frac = float(spatial_min_tissue_frac)
        self.spatial_min_signal = float(spatial_min_signal)
        if self.dab_nonnegative not in {"clamp", "softplus", "none"}:
            raise ValueError(
                "dab_nonnegative must be one of {'clamp', 'softplus', 'none'}"
            )
        if self.stain_reference_mode not in {"none", "batch_avg", "batch_median"}:
            raise ValueError(
                "stain_reference_mode must be one of {'none', 'batch_avg', 'batch_median'}"
            )
        if not (0.0 <= self.stain_ref_blend <= 1.0):
            raise ValueError("stain_ref_blend must be in [0, 1].")
        if self.stain_min_ref_images < 1:
            raise ValueError("stain_min_ref_images must be >= 1.")
        if self.spatial_weight < 0.0:
            raise ValueError("spatial_weight must be >= 0.")
        if self.spatial_grid_size < 1:
            raise ValueError("spatial_grid_size must be >= 1.")
        if not (0.0 <= self.spatial_min_tissue_frac <= 1.0):
            raise ValueError("spatial_min_tissue_frac must be in [0, 1].")
        if self.spatial_min_signal < 0.0:
            raise ValueError("spatial_min_signal must be >= 0.")

    def _tanh_to_01(self, x):
        """Convert from [-1, 1] (Tanh output) to [0, 1]."""
        return (x + 1.0) / 2.0

    def forward(self, y, y_hat):
        """Compute L_expr over a batch.

        Args:
            y: (B, 3, H, W) real IHC in [-1, 1].
            y_hat: (B, 3, H, W) generated IHC in [-1, 1].

        Returns:
            loss: scalar expression loss (mean over active images in batch).
            info: dict with global and spatial DAB values for logging.
        """
        y_01 = self._tanh_to_01(y)
        yhat_01 = self._tanh_to_01(y_hat)

        B = y.shape[0]

        # Tissue mask from real target only
        mask = tissue_mask_from_od(
            y_01.detach(), self.od_threshold, self.dilate_radius
        )  # (B, H, W)

        # Estimate stain matrices from real target only (detached)
        V_list = estimate_stain_matrix(
            y_01.detach(),
            self.od_threshold,
            reference_mode=self.stain_reference_mode,
            ref_blend=self.stain_ref_blend,
            min_ref_images=self.stain_min_ref_images,
        )

        # Accumulate losses into a list and stack at the end, avoiding repeated
        # creation of disconnected zero tensors in the loop.
        image_losses = []
        iod_real_sum, iod_gen_sum = 0.0, 0.0
        miod_real_sum, miod_gen_sum = 0.0, 0.0
        spatial_loss_sum = 0.0
        spatial_active_patch_frac_sum = 0.0

        for i in range(B):
            # No grad through stain estimation matrix.
            V = V_list[i].detach()  # (3, 2)

            # dab_y is from the real image (already detached); dab_yhat carries grad.
            dab_y = get_dab_map(
                y_01[i].detach(),
                V,
                nonnegative=self.dab_nonnegative,
                softplus_beta=self.softplus_beta,
            )
            dab_yhat = get_dab_map(
                yhat_01[i],
                V,
                nonnegative=self.dab_nonnegative,
                softplus_beta=self.softplus_beta,
            )

            m = mask[i].detach()  # (H, W)

            # iod_y / miod_y are already detached (computed from detached dab_y).
            iod_y, miod_y = _compute_iod_miod(dab_y, m)
            iod_yh, miod_yh = _compute_iod_miod(dab_yhat, m)

            global_terms = []

            # Skip low-signal targets to avoid outlier-dominated relative terms.
            if (iod_y >= self.min_iod_signal).item():
                loss_iod = _symmetric_relative_error(
                    iod_yh,
                    iod_y,
                    min_den=MIN_IOD_DEN,
                )
                global_terms.append(torch.clamp(loss_iod, max=self.max_rel_term))

            if (miod_y >= self.min_miod_signal).item():
                loss_miod = _symmetric_relative_error(
                    miod_yh,
                    miod_y,
                    min_den=MIN_MIOD_DEN,
                )
                global_terms.append(torch.clamp(loss_miod, max=self.max_rel_term))

            terms = []
            if global_terms:
                terms.append(torch.stack(global_terms).mean())

            spatial_loss_i_val = 0.0
            spatial_active_patch_frac_i = 0.0
            if self.spatial_weight > 0.0:
                spatial_loss_i, spatial_active_patch_frac_i = _compute_patchwise_relative_miod(
                    dab_y,
                    dab_yhat,
                    m,
                    self.spatial_grid_size,
                    self.spatial_min_tissue_frac,
                    self.spatial_min_signal,
                    self.max_rel_term,
                )
                spatial_loss_i_val = spatial_loss_i.item()
                if spatial_active_patch_frac_i > 0.0:
                    terms.append(self.spatial_weight * spatial_loss_i)

            if terms:
                image_losses.append(torch.stack(terms).sum())

            iod_real_sum += iod_y.item()
            iod_gen_sum += iod_yh.item()
            miod_real_sum += miod_y.item()
            miod_gen_sum += miod_yh.item()
            spatial_loss_sum += spatial_loss_i_val
            spatial_active_patch_frac_sum += spatial_active_patch_frac_i

        active_images = len(image_losses)
        if active_images > 0:
            loss = torch.stack(image_losses).mean()
        else:
            # Keep this zero connected to the graph so backward() is always valid,
            # even if ExpressionLoss is used standalone and all samples are inactive.
            loss = y_hat.sum() * 0.0

        info = {
            "iod_real": iod_real_sum / B,
            "iod_gen": iod_gen_sum / B,
            "miod_real": miod_real_sum / B,
            "miod_gen": miod_gen_sum / B,
            "expr_active_frac": active_images / B,
            "expr_spatial": spatial_loss_sum / B,
            "expr_spatial_active_patch_frac": spatial_active_patch_frac_sum / B,
        }
        return loss, info
