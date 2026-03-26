import cv2
import numpy as np
from scipy.optimize import least_squares


def project_points(H, pts):
    """
    Vectorized projection of Nx2 points using a 3x3 Homography Matrix.
    """
    pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    proj = (H @ pts_homo.T).T
    z = proj[:, 2:] + 1e-6
    proj_2d = proj[:, :2] / z
    return proj_2d


def homography_residuals_vectorized(h_elements, pts_A, means_B, inv_covs_B):
    """
    Fully vectorized Mahalanobis distance calculation using Einstein Summation.
    """
    H = np.append(h_elements, 1.0).reshape(3, 3)
    pts_B_proj = project_points(H, pts_A)
    err = pts_B_proj - means_B
    mahalanobis_sq = np.einsum("ni,nij,nj->n", err, inv_covs_B, err)
    return np.sqrt(np.maximum(mahalanobis_sq, 0))


def srt_to_matrix(params):
    """
    Build a 3x3 similarity (sRT) matrix from [s, theta, tx, ty].
    """
    s, theta, tx, ty = params
    s = np.clip(s, 0.0, 4.0)
    theta = np.clip(theta, np.radians(-90.0), np.radians(90.0))
    c, si = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [s * c, -s * si, tx],
            [s * si, s * c, ty],
            [0.0, 0.0, 1.0],
        ]
    )


def srt_residuals(
    params,
    pts_A,
    means_B,
    inv_covs_B,
    scale_reg_weight=0.01,
    rot_reg_weight=0.001,
):
    """
    Mahalanobis residuals for a similarity (sRT) homography parametrised as
    [s, theta, tx, ty].
    """
    H = srt_to_matrix(params)
    pts_B_proj = project_points(H, pts_A)
    err = pts_B_proj - means_B
    mahalanobis_sq = np.einsum("ni,nij,nj->n", err, inv_covs_B, err)
    mahalanobis_loss = np.sqrt(np.maximum(mahalanobis_sq, 0))

    scale_loss = (params[0] - 1.0) ** 2
    rot_loss = np.degrees(params[1]) ** 2
    regularization = scale_loss * scale_reg_weight + rot_loss * rot_reg_weight
    return mahalanobis_loss + regularization


def extract_srt_from_homography(H):
    """
    Extract the best-fit sRT parameters [s, theta, tx, ty] from a general
    3x3 homography by reading the top-left 2x2 block.
    """
    a = H[0, 0]
    b = H[1, 0]
    tx = H[0, 2]
    ty = H[1, 2]
    s = np.sqrt(a**2 + b**2)
    theta = np.arctan2(b, a)
    return np.array([s, theta, tx, ty])


def optimize_homography(
    pts_A,
    means_B,
    covs_B,
    peaks_B=None,
    model="full",
    use_means_for_ransac=False,
    verbose=0,
    quiet=False,
    ransac_method=cv2.RANSAC,
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
    Optimizes the homography using the Mahalanobis distance and Huber Loss.
    """
    N = pts_A.shape[0]
    if N < 4:
        raise ValueError("At least 4 points are required to compute a homography.")

    covs_safe = covs_B + np.eye(2) * 1e-6
    inv_covs_B = np.linalg.inv(covs_safe)

    if use_means_for_ransac:
        ransac_pts_B = means_B
    else:
        ransac_pts_B = peaks_B if peaks_B is not None else means_B

    H_init, inlier_mask = cv2.findHomography(
        pts_A,
        ransac_pts_B,
        ransac_method,
        ransacReprojThreshold=ransac_reproj_threshold,
        maxIters=ransac_max_iters,
        confidence=ransac_confidence,
    )
    if H_init is None:
        if not quiet:
            print("Warning: RANSAC failed to find an initial guess. Defaulting to Identity.")
        H_init = np.eye(3)
        inlier_mask = np.zeros((N, 1), dtype=np.uint8)

    H_init_norm = H_init / H_init[2, 2]
    h_init_elements = H_init_norm.flatten()[:8]

    if model == "sRT":
        srt_init = extract_srt_from_homography(H_init_norm)
        if not quiet:
            print(
                f"  sRT init: s={srt_init[0]:.4f}, θ={np.degrees(srt_init[1]):.2f}°, "
                f"tx={srt_init[2]:.2f}, ty={srt_init[3]:.2f}"
            )

        res = least_squares(
            fun=srt_residuals,
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
            fun=homography_residuals_vectorized,
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
    Computes the mean L2 corner error between two homographies
    given the width and height of the referenced image.
    """
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    corners_gt = project_points(H_gt, corners)
    corners_pred = project_points(H_pred, corners)
    error = np.linalg.norm(corners_gt - corners_pred, axis=1).mean()
    return error
