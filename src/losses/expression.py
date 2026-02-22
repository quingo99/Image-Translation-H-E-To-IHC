"""Expression loss L_expr: clinically guided DAB expression intensity agreement.

L_expr = |IOD(y_hat) - IOD(y)| / (IOD(y) + eps)
       + |mIOD(y_hat) - mIOD(y)| / (mIOD(y) + eps)

where:
    IOD(I)  = sum_p m_p * D_p(I)              (total DAB on tissue)
    mIOD(I) = sum_p m_p * D_p(I) / (sum_p m_p + eps)  (mean DAB on tissue)
"""

import torch
import torch.nn as nn

from src.stains.macenko import estimate_stain_matrix, get_dab_map, tissue_mask_from_od

EPS = 1e-6
MIN_IOD_DEN = 1e-2
MIN_MIOD_DEN = 1e-3
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


class ExpressionLoss(nn.Module):
    """Clinically guided expression loss using Macenko stain separation.

    Computes relative error between IOD/mIOD of real and generated images.
    """

    def __init__(
        self,
        od_threshold=0.15,
        dilate_radius=3,
        min_iod_signal=0.0,
        min_miod_signal=0.0,
        max_rel_term=DEFAULT_MAX_REL_TERM,
        dab_nonnegative="clamp",
        softplus_beta=10.0,
        stain_reference_mode="none",
        stain_ref_blend=0.0,
        stain_min_ref_images=4,
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
        if self.dab_nonnegative not in {"clamp", "softplus", "none"}:
            raise ValueError(
                "dab_nonnegative must be one of {'clamp', 'softplus', 'none'}"
            )
        if self.stain_reference_mode not in {"none", "batch_avg"}:
            raise ValueError(
                "stain_reference_mode must be one of {'none', 'batch_avg'}"
            )
        if not (0.0 <= self.stain_ref_blend <= 1.0):
            raise ValueError("stain_ref_blend must be in [0, 1].")
        if self.stain_min_ref_images < 1:
            raise ValueError("stain_min_ref_images must be >= 1.")

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
            info: dict with IOD/mIOD values for logging.
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

        total_loss = torch.tensor(0.0, device=y.device)
        active_images = 0
        iod_real_sum, iod_gen_sum = 0.0, 0.0
        miod_real_sum, miod_gen_sum = 0.0, 0.0

        for i in range(B):
            # No grad through stain estimation matrix.
            V = V_list[i].detach()  # (3, 2)

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

            m = mask[i]  # (H, W)

            iod_y, miod_y = _compute_iod_miod(dab_y, m.detach())
            iod_yh, miod_yh = _compute_iod_miod(dab_yhat, m.detach())

            image_loss = torch.tensor(0.0, device=y.device)
            n_terms = 0

            # Skip low-signal targets to avoid outlier-dominated relative terms.
            if bool((iod_y.detach() >= self.min_iod_signal).item()):
                den_iod = torch.clamp(iod_y.detach(), min=MIN_IOD_DEN)
                loss_iod = torch.abs(iod_yh - iod_y) / den_iod
                loss_iod = torch.clamp(loss_iod, max=self.max_rel_term)
                image_loss = image_loss + loss_iod
                n_terms += 1

            if bool((miod_y.detach() >= self.min_miod_signal).item()):
                den_miod = torch.clamp(miod_y.detach(), min=MIN_MIOD_DEN)
                loss_miod = torch.abs(miod_yh - miod_y) / den_miod
                loss_miod = torch.clamp(loss_miod, max=self.max_rel_term)
                image_loss = image_loss + loss_miod
                n_terms += 1

            if n_terms > 0:
                total_loss = total_loss + image_loss / n_terms
                active_images += 1

            iod_real_sum += iod_y.item()
            iod_gen_sum += iod_yh.item()
            miod_real_sum += miod_y.item()
            miod_gen_sum += miod_yh.item()

        if active_images > 0:
            loss = total_loss / active_images
        else:
            loss = torch.tensor(0.0, device=y.device)

        info = {
            "iod_real": iod_real_sum / B,
            "iod_gen": iod_gen_sum / B,
            "miod_real": miod_real_sum / B,
            "miod_gen": miod_gen_sum / B,
            "expr_active_frac": active_images / B,
        }
        return loss, info
