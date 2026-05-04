"""RANSAC initialization helper module.

Centralizes cv2.findHomography initialization logic used by all backends.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def ransac_init(
    pts_A: np.ndarray,
    ransac_pts_B: np.ndarray,
    method: int = cv2.USAC_FAST,
    reproj_threshold: float = 3.0,
    max_iters: int = 5000,
    confidence: float = 0.995,
    quiet: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize homography using cv2.findHomography with RANSAC.

    Args:
        pts_A: (N, 2) points in image A
        ransac_pts_B: (N, 2) points in image B (peaks or means)
        method: cv2 RANSAC method (USAC_FAST, RANSAC, etc.)
        reproj_threshold: Reprojection error threshold
        max_iters: Maximum iterations
        confidence: Confidence level (0.0-1.0)
        quiet: If True, suppress warnings

    Returns:
        (H_init, H_init_norm): H_init from cv2, H_init_norm = H_init / H_init[2, 2]
    """
    H_init, inlier_mask = cv2.findHomography(
        pts_A,
        ransac_pts_B,
        method,
        ransacReprojThreshold=reproj_threshold,
        maxIters=max_iters,
        confidence=confidence,
    )

    if H_init is None:
        if not quiet:
            print("Warning: RANSAC failed to find an initial guess. Defaulting to Identity.")
        H_init = np.eye(3)

    H_init_norm = H_init / H_init[2, 2]
    return H_init, H_init_norm
