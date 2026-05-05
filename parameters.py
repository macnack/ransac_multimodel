"""Centralized parameter structures for homography optimization.

Replaces scattered kwargs with typed dataclasses for clarity,
validation, and consistency across all backends (numpy, torch, theseus).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass
class HomographyParams:
    """
    Parameterization and bounds for homography or sRT model.

    Attributes:
        model: "full" (8-DOF homography) or "sRT" (4-DOF similarity)
        bounds_low: Lower bounds for optimization (shape: (4,) or (8,))
        bounds_high: Upper bounds for optimization (shape: (4,) or (8,))
        x_scale: Step size scaling hints for optimizer (sRT only)
    """

    model: Literal["full", "sRT"] = "full"
    bounds_low: Optional[np.ndarray] = None
    bounds_high: Optional[np.ndarray] = None
    x_scale: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate and set defaults based on model."""
        if self.model == "sRT":
            if self.bounds_low is None:
                self.bounds_low = np.array([0.25, np.radians(-180), -60, -60])
            if self.bounds_high is None:
                self.bounds_high = np.array([3.0, np.radians(180), 60, 60])
            if self.x_scale is None:
                self.x_scale = np.array([0.20, 1.0, 1.0, 1.0])
        else:
            if self.bounds_low is not None or self.bounds_high is not None:
                raise ValueError("bounds_low/high not supported for 'full' model")
            if self.x_scale is not None:
                raise ValueError("x_scale not supported for 'full' model")


@dataclass
class RegularizationParams:
    """
    Regularization weights and parameters.

    Attributes:
        scale_reg_weight: Penalty weight for scale deviations from 1.0 (sRT only)
        rot_reg_weight: Penalty weight for rotation deviations from 0 (sRT only)
    """

    scale_reg_weight: float = 0.01
    rot_reg_weight: float = 0.001


@dataclass
class OptimizationConfig:
    """
    Optimizer configuration shared across backends.

    Attributes:
        max_nfev: Maximum number of function evaluations (scipy) or iterations (torch)
        robust_loss: Type of robust loss ("huber", "soft_l1", etc.)
        f_scale: Transition point for Huber loss
        verbose: Verbosity level (0=silent, 1+= increasingly verbose)
        quiet: If True, suppress print statements
    """

    max_nfev: int = 5000
    robust_loss: str = "huber"
    f_scale: float = 2.0
    verbose: int = 0
    quiet: bool = False


@dataclass
class RansacConfig:
    """
    RANSAC initialization configuration (cv2.findHomography).

    Attributes:
        method: cv2 method (RANSAC, USAC_FAST, etc.)
        reproj_threshold: Reprojection threshold (pixels)
        max_iters: Maximum iterations
        confidence: Confidence level (0.0-1.0)
        use_means_for_ransac: If False, use peaks_B; if True, use means_B
    """

    method: int = None  # Will be set to cv2.USAC_FAST by default
    reproj_threshold: float = 3.0
    max_iters: int = 5000
    confidence: float = 0.995
    use_means_for_ransac: bool = False

    def __post_init__(self):
        """Set default method."""
        if self.method is None:
            import cv2

            self.method = cv2.USAC_FAST


@dataclass
class TorchOptimizationConfig:
    """Torch-specific optimization settings (LBFGS, LM, etc.)."""

    device: str = "cpu"
    dtype: str = "float64"
    lr: float = 1.0
    max_iter: int = 200
    tolerance_grad: float = 1e-12
    tolerance_change: float = 1e-12


@dataclass
class TorchLMConfig:
    """Levenberg-Marquardt specific configuration (homography_torch_lm.py)."""

    device: str = "cpu"
    dtype: str = "float64"
    max_lm_iter: int = 20
    lm_damping_init: float = 1e-4
    lm_damping_down: float = 0.7
    lm_damping_up: float = 10.0
    lm_damping_max: float = 1e8
    converge_rtol: float = 1e-4
    barrier_scale: float = 100.0


def create_default_homography_params(
    model: Literal["full", "sRT"] = "full",
) -> HomographyParams:
    """Factory function for default HomographyParams."""
    return HomographyParams(model=model)


def create_default_ransac_config() -> RansacConfig:
    """Factory function for default RansacConfig."""
    return RansacConfig()
