import numpy as np

from .gaussian_extraction import (
    softmax_heatmaps,
    process_patches,
    assemble_correspondences,
)


def find_gaussians(
    tensor,
    adaptive_gauss_fit: bool = True,
    plot_heatmaps: bool = False,
    plotter=None,
    log_missing_gaussians: bool = True,
    missing_gaussians_logger=None,
    adaptive_threshold: float = 0.003,
    adaptive_n_sigma: float = 3.0,
    adaptive_max_iter: int = 10,
    adaptive_min_half_w: int = 1,
    adaptive_max_half_w: int = 5,
    fixed_threshold: float = 0.008,
    fixed_window_size: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit Gaussians to every patch heatmap in *tensor* and return correspondences.
    """
    # Apply softmax to tensor and extract patch dimensions
    tensor_softmax, out_patch_size = softmax_heatmaps(tensor)

    # Extract Gaussians per patch
    dict_of_gaussians = process_patches(
        tensor_softmax,
        adaptive_gauss_fit=adaptive_gauss_fit,
        adaptive_threshold=adaptive_threshold,
        adaptive_n_sigma=adaptive_n_sigma,
        adaptive_max_iter=adaptive_max_iter,
        adaptive_min_half_w=adaptive_min_half_w,
        adaptive_max_half_w=adaptive_max_half_w,
        fixed_threshold=fixed_threshold,
        fixed_window_size=fixed_window_size,
        log_missing_gaussians=False,  # Will handle logging after plotting
        missing_gaussians_logger=None,
    )

    # Handle optional plotting (before logging missing Gaussians)
    if plot_heatmaps and plotter is not None:
        for (px, py), gaussians in dict_of_gaussians.items():
            heatmap = tensor_softmax[:, py, px].reshape(out_patch_size, out_patch_size)
            plotter(heatmap, gaussians, px, py)

    # Assemble correspondences from Gaussian dict
    pts_A, means_B, peaks_B, covs_B = assemble_correspondences(
        dict_of_gaussians,
        log_missing_gaussians=log_missing_gaussians,
        missing_gaussians_logger=missing_gaussians_logger,
    )

    return pts_A, means_B, peaks_B, covs_B
