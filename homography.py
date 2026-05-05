import numpy as np
from scipy.optimize import least_squares

from .geometry_utils import (
    extract_srt_from_homography,
    project_points,
    srt_to_matrix,
)
from .ransac_init import ransac_init
from .residuals import (
    huber_loss,
    mahalanobis_residuals,
    regularization_loss,
)


def homography_residuals_vectorized(h_elements, pts_A, means_B, inv_covs_B):
    """Residuals for full 8-DOF homography model.

    Fully vectorized Mahalanobis distance calculation. Public symbol used by
    parity tests (``tests/test_torch_parity.py``) and external callers.
    """
    H = np.append(h_elements, 1.0).reshape(3, 3)
    pts_B_proj = project_points(H, pts_A)
    err = pts_B_proj - means_B
    return mahalanobis_residuals(err, inv_covs_B)


# Backwards-compatible private alias retained for internal references below.
_homography_residuals_full = homography_residuals_vectorized


def _srt_residuals_full(
    params,
    pts_A,
    means_B,
    inv_covs_B,
    scale_reg_weight=0.01,
    rot_reg_weight=0.001,
):
    """Residuals + regularization for sRT 4-DOF model."""
    H = srt_to_matrix(params)
    pts_B_proj = project_points(H, pts_A)
    err = pts_B_proj - means_B
    mahalanobis_loss = mahalanobis_residuals(err, inv_covs_B)
    regularization = regularization_loss(params, model="sRT", scale_reg_weight=scale_reg_weight, rot_reg_weight=rot_reg_weight)
    return mahalanobis_loss + regularization


def optimize_homography(
    pts_A,
    means_B,
    covs_B,
    peaks_B=None,
    model="full",
    use_means_for_ransac=False,
    verbose=0,
    quiet=False,
    ransac_method=None,
    ransac_reproj_threshold=3.0,
    ransac_max_iters=5000,
    ransac_confidence=0.995,
    robust_loss="huber",
    f_scale=2.0,
    max_nfev=5000,
    srt_x_scale=(0.20, 1.0, 1.0, 1.0),
    srt_bounds=([0.25, np.radians(-180), -60, -60], [3.0, np.radians(180), 60, 60]),
    srt_scale_reg_weight=0.01,
    srt_rot_reg_weight=0.001,
    return_details=False,
):
    """
    Optimize the homography using Mahalanobis distance and robust loss.
    
    Uses RANSAC for initialization, then refines with scipy.optimize.least_squares.
    """
    if ransac_method is None:
        import cv2
        ransac_method = cv2.USAC_FAST

    N = pts_A.shape[0]
    if N < 4:
        raise ValueError("At least 4 points are required to compute a homography.")

    covs_safe = covs_B + np.eye(2) * 1e-6
    inv_covs_B = np.linalg.inv(covs_safe)

    if use_means_for_ransac:
        ransac_pts_B = means_B
    else:
        ransac_pts_B = peaks_B if peaks_B is not None else means_B

    H_init, H_init_norm = ransac_init(
        pts_A,
        ransac_pts_B,
        method=ransac_method,
        reproj_threshold=ransac_reproj_threshold,
        max_iters=ransac_max_iters,
        confidence=ransac_confidence,
        quiet=quiet,
    )
    inlier_mask = np.zeros((N, 1), dtype=np.uint8)

    h_init_elements = H_init_norm.flatten()[:8]

    if model == "sRT":
        srt_init = extract_srt_from_homography(H_init_norm)
        if not quiet:
            print(
                f"  sRT init: s={srt_init[0]:.4f}, θ={np.degrees(srt_init[1]):.2f}°, "
                f"tx={srt_init[2]:.2f}, ty={srt_init[3]:.2f}"
            )

        res = least_squares(
            fun=_srt_residuals_full,
            x0=srt_init,
            args=(pts_A, means_B, inv_covs_B, srt_scale_reg_weight, srt_rot_reg_weight),
            method="trf",
            loss=robust_loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
            verbose=verbose,
            x_scale=srt_x_scale,
            bounds=srt_bounds,
        )

        if not quiet:
            print(
                f"  sRT optimized: s={res.x[0]:.4f}, θ={np.degrees(res.x[1]):.2f}°, "
                f"tx={res.x[2]:.2f}, ty={res.x[3]:.2f}"
            )
        H_opt = srt_to_matrix(res.x)
    else:
        res = least_squares(
            fun=_homography_residuals_full,
            x0=h_init_elements,
            args=(pts_A, means_B, inv_covs_B),
            method="trf",
            loss=robust_loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
            verbose=verbose,
        )
        H_opt = np.append(res.x, 1.0).reshape(3, 3)

    if return_details:
        details = {
            "num_correspondences": int(N),
            "num_inliers": int(np.sum(inlier_mask)) if inlier_mask is not None else 0,
            "inlier_ratio": float(np.mean(inlier_mask)) if inlier_mask is not None else 0.0,
            "optimization_success": bool(getattr(res, "success", True)),
            "optimization_status": int(getattr(res, "status", 0)),
            "optimization_message": str(getattr(res, "message", "")),
            "optimization_cost": float(getattr(res, "cost", np.nan)),
            "optimization_nfev": int(getattr(res, "nfev", 0)),
        }
        return H_opt, H_init, details

    return H_opt, H_init


def compute_corner_error(H_gt, H_pred, w=1024, h=1024):
    """
    Compute mean L2 corner error between two homographies
    given the width and height of the referenced image.
    
    Delegates to geometry_utils for the actual computation.
    """
    from .geometry_utils import compute_corner_error as _compute_corner_error
    return _compute_corner_error(H_gt, H_pred, w, h)

