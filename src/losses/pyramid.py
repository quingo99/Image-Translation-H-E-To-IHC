"""Gaussian Pyramid L1 loss for misalignment-tolerant reconstruction.

L_pyr(y, y_hat) = sum_{s=0}^{S} w_s * ||P_s(y) - P_s(y_hat)||_1

Uses separable 5x5 kernel [1,4,6,4,1]/16 and stride=2 downsampling.
Weights w_s = 1/2^s (fixed, not tuned).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPyramid(nn.Module):
    """Build a Gaussian pyramid from a tensor image."""

    def __init__(self, levels: int = 3):
        super().__init__()
        self.levels = levels
        # Separable 5x5 Gaussian kernel: [1,4,6,4,1]/16
        k1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)  # outer product -> 5x5
        # Shape: (1, 1, 5, 5) — applied per-channel with groups
        self.register_buffer("kernel", k2d.unsqueeze(0).unsqueeze(0))

    def _blur_and_down(self, x):
        """Apply Gaussian blur then downsample by 2."""
        B, C, H, W = x.shape
        # Expand kernel to (C, 1, 5, 5) for depthwise convolution
        k = self.kernel.expand(C, -1, -1, -1)
        blurred = F.conv2d(x, k, stride=1, padding=2, groups=C)
        return blurred[:, :, ::2, ::2]  # stride-2 subsample

    def forward(self, x):
        """Return list of pyramid levels [P_0, P_1, ..., P_S]."""
        levels = [x]
        current = x
        for _ in range(self.levels):
            current = self._blur_and_down(current)
            levels.append(current)
        return levels


class PyramidL1Loss(nn.Module):
    """Multi-scale Gaussian pyramid L1 loss.

    Args:
        levels: number of pyramid levels S (total S+1 including original).
    """

    def __init__(self, levels: int = 3):
        super().__init__()
        self.pyramid = GaussianPyramid(levels=levels)
        # Weights w_s = 1/2^s
        self.register_buffer(
            "weights",
            torch.tensor([1.0 / (2.0 ** s) for s in range(levels + 1)]),
        )

    def forward(self, y, y_hat):
        """Compute L_pyr(y, y_hat)."""
        pyr_y = self.pyramid(y)
        pyr_yhat = self.pyramid(y_hat)

        loss = torch.tensor(0.0, device=y.device)
        for s, (py, pyh) in enumerate(zip(pyr_y, pyr_yhat)):
            loss = loss + self.weights[s] * F.l1_loss(py, pyh)
        return loss
