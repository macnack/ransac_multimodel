"""Shared Gaussian extraction helpers.

Factors out common logic from correspondence.py and correspondence_torch.py
to reduce duplication and improve maintainability.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def softmax_heatmaps(
    tensor: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Apply softmax to tensor and extract patch dimensions.

    Args:
        tensor: (M, H, W) logits where M = out_patch_dim^2

    Returns:
        (tensor_softmax, out_patch_dim)
            tensor_softmax: (M, H, W) with softmax applied to M dimension
            out_patch_dim: int, dimension of patch grid
    """
    import torch

    out_patch_size = int(tensor.shape[0] ** 0.5)
    assert (
        out_patch_size * out_patch_size == tensor.shape[0]
    ), "Tensor's first dimension must be a perfect square."

    tensor_softmax = torch.softmax(tensor.float(), dim=0).numpy()
    return tensor_softmax, out_patch_size


def process_patches(
    tensor: np.ndarray,
    adaptive_gauss_fit: bool = True,
    adaptive_threshold: float = 0.003,
    adaptive_n_sigma: float = 3.0,
    adaptive_max_iter: int = 10,
    adaptive_min_half_w: int = 1,
    adaptive_max_half_w: int = 5,
    fixed_threshold: float = 0.008,
    fixed_window_size: int = 4,
    log_missing_gaussians: bool = True,
    missing_gaussians_logger: Optional[callable] = None,
) -> dict:
    """
    Process every (py, px) patch and extract Gaussians.

    Args:
        tensor: (M, H, W) softmax-normalized heatmaps
        adaptive_gauss_fit: Use adaptive or fixed window extraction
        adaptive_threshold: Threshold for adaptive extraction
        adaptive_n_sigma: Number of sigmas for adaptive window expansion
        adaptive_max_iter: Max iterations for window adaptation
        adaptive_min_half_w: Minimum half-window size
        adaptive_max_half_w: Maximum half-window size
        fixed_threshold: Threshold for fixed extraction
        fixed_window_size: Fixed window size
        log_missing_gaussians: Print when no Gaussians found in patch
        missing_gaussians_logger: Callback for missing Gaussians

    Returns:
        {(px, py): [gaussians]} dict mapping patch coords to Gaussian lists
    """
    from .gaussian_fit import (
        extract_gaussians_adaptive,
        extract_gaussians_from_heatmap2,
    )

    tensor_softmax, out_patch_size = softmax_heatmaps(tensor)
    dict_of_gaussians = {}

    for py in range(tensor.shape[1]):
        for px in range(tensor.shape[2]):
            heatmap = tensor_softmax[:, py, px]
            heatmap = heatmap.reshape(out_patch_size, out_patch_size)

            if adaptive_gauss_fit:
                gaussians = extract_gaussians_adaptive(
                    heatmap,
                    threshold=adaptive_threshold,
                    n_sigma=adaptive_n_sigma,
                    max_iter=adaptive_max_iter,
                    min_half_w=adaptive_min_half_w,
                    max_half_w=adaptive_max_half_w,
                )
            else:
                gaussians = extract_gaussians_from_heatmap2(
                    heatmap,
                    threshold=fixed_threshold,
                    window_size=fixed_window_size,
                )

            coord = (px, py)
            dict_of_gaussians[coord] = gaussians

    return dict_of_gaussians


def assemble_correspondences(
    dict_of_gaussians: dict,
    log_missing_gaussians: bool = True,
    missing_gaussians_logger: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble correspondence arrays from Gaussian dict.

    Args:
        dict_of_gaussians: {(px, py): [gaussians]} dict
        log_missing_gaussians: Print when no Gaussians found in patch
        missing_gaussians_logger: Callback for missing Gaussians

    Returns:
        (pts_A, means_B, peaks_B, covs_B)
    """
    correspondences_A = []
    correspondences_B_mu = []
    correspondences_B_peak = []
    correspondences_B_cov = []

    for (px, py), gaussians in dict_of_gaussians.items():
        if not gaussians:
            if log_missing_gaussians:
                print(f"No Gaussians found in patch ({px}, {py})")
            if missing_gaussians_logger is not None:
                missing_gaussians_logger(px, py)
            continue

        for g in gaussians:
            lx, ly = g["mean"]
            correspondences_A.append([float(px), float(py)])
            correspondences_B_mu.append([lx, ly])

            peak = g["global_peak"]
            correspondences_B_peak.append([float(peak[0]), float(peak[1])])
            correspondences_B_cov.append(np.array(g["cov"]))

    pts_A = np.array(correspondences_A, dtype=np.float32)
    means_B = np.array(correspondences_B_mu, dtype=np.float32)
    peaks_B = np.array(correspondences_B_peak, dtype=np.float32)
    covs_B = np.array(correspondences_B_cov, dtype=np.float32)

    return pts_A, means_B, peaks_B, covs_B
