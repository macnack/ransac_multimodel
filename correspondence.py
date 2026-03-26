import numpy as np

from .gaussian_fit import extract_gaussians_adaptive, extract_gaussians_from_heatmap2


def find_gaussians(
    tensor,
    adaptive_gauss_fit: bool = True,
    plot_heatmaps: bool = False,
    plotter=None,
    log_missing_gaussians: bool = True,
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
    import torch

    dict_of_gaussians = {}
    out_patch_size = int(tensor.shape[0] ** 0.5)
    assert out_patch_size * out_patch_size == tensor.shape[0], (
        "Tensor's first dimension must be a perfect square."
    )

    for py in range(tensor.shape[1]):
        for px in range(tensor.shape[2]):
            heatmap = tensor[:, py, px]
            heatmap = torch.softmax(heatmap.float(), dim=0).numpy()
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

            if plot_heatmaps and plotter is not None:
                plotter(heatmap, gaussians, px, py)

    correspondences_A = []
    correspondences_B_mu = []
    correspondences_B_peak = []
    correspondences_B_cov = []

    for (px, py), gaussians in dict_of_gaussians.items():
        if not gaussians:
            if log_missing_gaussians:
                print(f"No Gaussians found in patch ({px}, {py})")
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
