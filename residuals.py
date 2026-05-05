"""Unified residual and loss calculation for homography optimization.

Provides backend-agnostic (numpy/torch) computation of:
- Mahalanobis distance residuals
- Huber robust loss
- Regularization terms (scale, rotation)
- Combined loss for optimization

This module eliminates duplication across homography.py, homography_torch.py,
and homography_torch_lm.py.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# NumPy Backend
# ============================================================================

def mahalanobis_residuals(
    err: np.ndarray,
    inv_covs: np.ndarray,
) -> np.ndarray:
    """
    Compute Mahalanobis distance residuals using vectorized einsum.

    Args:
        err: (N, 2) array of residuals (projected - target)
        inv_covs: (N, 2, 2) array of inverse covariance matrices

    Returns:
        (N,) array of Mahalanobis distances (square root of squared distances)
    """
    mahalanobis_sq = np.einsum("ni,nij,nj->n", err, inv_covs, err)
    return np.sqrt(np.maximum(mahalanobis_sq, 0.0))


def huber_loss(residuals: np.ndarray, f_scale: float = 2.0) -> np.ndarray:
    """
    Huber robust loss function.

    Args:
        residuals: (N,) or scalar array of residuals
        f_scale: transition point between quadratic and linear regimes

    Returns:
        (N,) or scalar array of losses
    """
    abs_r = np.abs(residuals)
    quad = 0.5 * residuals**2
    lin = f_scale * (abs_r - 0.5 * f_scale)
    return np.where(abs_r <= f_scale, quad, lin)


def regularization_loss(
    params: np.ndarray,
    model: str = "full",
    scale_reg_weight: float = 0.01,
    rot_reg_weight: float = 0.001,
) -> float:
    """
    Compute regularization loss for sRT parameters.

    Args:
        params: For sRT model: [s, theta, tx, ty]; for full model: unused
        model: "sRT" or "full"
        scale_reg_weight: weight for scale penalty (sRT only)
        rot_reg_weight: weight for rotation penalty (sRT only)

    Returns:
        Scalar regularization loss
    """
    if model != "sRT":
        return 0.0

    s, theta = params[0], params[1]
    scale_loss = (s - 1.0) ** 2
    rot_loss = np.degrees(theta) ** 2
    return float(scale_loss * scale_reg_weight + rot_loss * rot_reg_weight)


# ============================================================================
# Torch Backend
# ============================================================================

if HAS_TORCH:

    def mahalanobis_residuals_torch(
        err: torch.Tensor,
        inv_covs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Torch version of mahalanobis_residuals.

        Args:
            err: (N, 2) or (B, N, 2) tensor of residuals
            inv_covs: (N, 2, 2) or (B, N, 2, 2) tensor of inverse covariances

        Returns:
            (N,) or (B, N) tensor of Mahalanobis distances
        """
        mahalanobis_sq = torch.einsum("...ni,...nij,...nj->...n", err, inv_covs, err)
        return torch.sqrt(torch.clamp(mahalanobis_sq, min=0.0))

    def huber_loss_torch(residuals: torch.Tensor, f_scale: float = 2.0) -> torch.Tensor:
        """
        Torch version of huber_loss.

        Args:
            residuals: scalar or (N,) or (B, N) tensor
            f_scale: transition point

        Returns:
            Scalar or (N,) or (B, N) tensor of losses
        """
        abs_r = torch.abs(residuals)
        quad = 0.5 * residuals**2
        lin = f_scale * (abs_r - 0.5 * f_scale)
        return torch.where(abs_r <= f_scale, quad, lin)

    def regularization_loss_torch(
        params: torch.Tensor,
        model: str = "full",
        scale_reg_weight: float = 0.01,
        rot_reg_weight: float = 0.001,
    ) -> torch.Tensor:
        """
        Torch version of regularization_loss (sRT only).

        Args:
            params: For sRT: (4,) or (B, 4) tensor [s, theta, tx, ty]
            model: "sRT" or "full"
            scale_reg_weight: scale penalty weight
            rot_reg_weight: rotation penalty weight

        Returns:
            Scalar tensor (per-batch if (B, 4) input)
        """
        if model != "sRT":
            return params.new_tensor(0.0)

        # Handle both (4,) and (B, 4) shapes
        if params.dim() == 1:
            s = params[0]
            theta = params[1]
        else:
            s = params[..., 0]
            theta = params[..., 1]

        scale_loss = (s - 1.0) ** 2
        rot_loss = torch.rad2deg(theta) ** 2
        return scale_loss * scale_reg_weight + rot_loss * rot_reg_weight


# ============================================================================
# Unified Interface (Backend Selection)
# ============================================================================

def mahalanobis_residuals_unified(
    err: Union[np.ndarray, "torch.Tensor"],
    inv_covs: Union[np.ndarray, "torch.Tensor"],
) -> Union[np.ndarray, "torch.Tensor"]:
    """Backend-agnostic Mahalanobis residual computation."""
    if HAS_TORCH and isinstance(err, torch.Tensor):
        return mahalanobis_residuals_torch(err, inv_covs)
    return mahalanobis_residuals(err, inv_covs)


def huber_loss_unified(
    residuals: Union[np.ndarray, "torch.Tensor"],
    f_scale: float = 2.0,
) -> Union[np.ndarray, "torch.Tensor"]:
    """Backend-agnostic Huber loss computation."""
    if HAS_TORCH and isinstance(residuals, torch.Tensor):
        return huber_loss_torch(residuals, f_scale)
    return huber_loss(residuals, f_scale)
