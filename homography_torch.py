import cv2
import numpy as np
import torch

from .parity_utils import np_to_torch, torch_to_np


def project_points_torch(H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """
    Torch equivalent of project_points using homogeneous coordinates.
    """
    ones = torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)
    pts_homo = torch.cat([pts, ones], dim=1)
    proj = (H @ pts_homo.T).T
    z = proj[:, 2:] + 1e-6
    proj_2d = proj[:, :2] / z
    return proj_2d


def homography_residuals_vectorized_torch(
    h_elements: torch.Tensor,
    pts_A: torch.Tensor,
    means_B: torch.Tensor,
    inv_covs_B: torch.Tensor,
) -> torch.Tensor:
    """
    Torch equivalent of vectorized Mahalanobis residuals.
    """
    H = torch.cat([h_elements, h_elements.new_tensor([1.0])]).view(3, 3)
    pts_B_proj = project_points_torch(H, pts_A)
    err = pts_B_proj - means_B
    mahalanobis_sq = torch.einsum("ni,nij,nj->n", err, inv_covs_B, err)
    return torch.sqrt(torch.clamp(mahalanobis_sq, min=0.0))


def srt_to_matrix_torch(params: torch.Tensor) -> torch.Tensor:
    """
    Build 3x3 similarity matrix from [s, theta, tx, ty].
    """
    s, theta, tx, ty = params[0], params[1], params[2], params[3]

    s = torch.clamp(s, 0.0, 4.0)
    theta = torch.clamp(theta, float(np.radians(-90.0)), float(np.radians(90.0)))

    c, si = torch.cos(theta), torch.sin(theta)
    return torch.stack(
        [
            torch.stack([s * c, -s * si, tx]),
            torch.stack([s * si, s * c, ty]),
            torch.stack(
                [
                    params.new_tensor(0.0),
                    params.new_tensor(0.0),
                    params.new_tensor(1.0),
                ]
            ),
        ]
    )


def srt_residuals_torch(
    params: torch.Tensor,
    pts_A: torch.Tensor,
    means_B: torch.Tensor,
    inv_covs_B: torch.Tensor,
) -> torch.Tensor:
    """
    Torch equivalent of sRT residual computation.
    """
    H = srt_to_matrix_torch(params)
    pts_B_proj = project_points_torch(H, pts_A)
    err = pts_B_proj - means_B

    mahalanobis_sq = torch.einsum("ni,nij,nj->n", err, inv_covs_B, err)
    mahalanobis_loss = torch.sqrt(torch.clamp(mahalanobis_sq, min=0.0))

    scale_loss = (params[0] - 1.0) ** 2
    rot_loss = torch.rad2deg(params[1]) ** 2
    regularization = scale_loss * 0.01 + rot_loss * 0.001

    return mahalanobis_loss + regularization


def _huber_cost(residuals: torch.Tensor, f_scale: float = 2.0) -> torch.Tensor:
    abs_r = torch.abs(residuals)
    quad = 0.5 * residuals**2
    lin = f_scale * (abs_r - 0.5 * f_scale)
    return torch.where(abs_r <= f_scale, quad, lin).mean()


def optimize_homography_torch(
    pts_A,
    means_B,
    covs_B,
    peaks_B=None,
    model="full",
    use_means_for_ransac=False,
    verbose=0,
    device="cpu",
    dtype=torch.float64,
    f_scale=2.0,
    max_iter=200,
    lr=1.0,
    quiet=False,
):
    """
    Torch optimizer path parallel to optimize_homography.

    Keeps OpenCV RANSAC initialization for parity and optimizes using LBFGS.
    Returns (H_opt, H_init) as numpy arrays.
    """
    pts_A = np.asarray(pts_A)
    means_B = np.asarray(means_B)
    covs_B = np.asarray(covs_B)

    N = pts_A.shape[0]
    if N < 4:
        raise ValueError("At least 4 points are required to compute a homography.")

    covs_safe = covs_B + np.eye(2) * 1e-6
    inv_covs_B_np = np.linalg.inv(covs_safe)

    if use_means_for_ransac:
        ransac_pts_B = means_B
    else:
        ransac_pts_B = peaks_B if peaks_B is not None else means_B

    H_init, _ = cv2.findHomography(pts_A, ransac_pts_B, cv2.RANSAC, maxIters=5000)
    if H_init is None:
        if not quiet:
            print("Warning: RANSAC failed to find an initial guess. Defaulting to Identity.")
        H_init = np.eye(3)

    H_init_norm = H_init / H_init[2, 2]

    pts_A_t = np_to_torch(pts_A, device=device, dtype=dtype)
    means_B_t = np_to_torch(means_B, device=device, dtype=dtype)
    inv_covs_B_t = np_to_torch(inv_covs_B_np, device=device, dtype=dtype)

    if model == "sRT":
        a = H_init_norm[0, 0]
        b = H_init_norm[1, 0]
        tx = H_init_norm[0, 2]
        ty = H_init_norm[1, 2]
        s = np.sqrt(a**2 + b**2)
        theta = np.arctan2(b, a)
        init = np.array([s, theta, tx, ty], dtype=np.float64)

        params = np_to_torch(init, device=device, dtype=dtype).clone().detach().requires_grad_(True)

        def residual_fn():
            return srt_residuals_torch(params, pts_A_t, means_B_t, inv_covs_B_t)

    else:
        h_init_elements = H_init_norm.flatten()[:8].astype(np.float64)
        params = np_to_torch(h_init_elements, device=device, dtype=dtype).clone().detach().requires_grad_(True)

        def residual_fn():
            return homography_residuals_vectorized_torch(params, pts_A_t, means_B_t, inv_covs_B_t)

    optimizer = torch.optim.LBFGS(
        [params],
        lr=lr,
        max_iter=max_iter,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        residuals = residual_fn()
        loss = _huber_cost(residuals, f_scale=f_scale)
        loss.backward()
        return loss

    final_loss = optimizer.step(closure)

    if verbose:
        print(f"  torch optimize done: loss={float(final_loss):.6f}")

    with torch.no_grad():
        if model == "sRT":
            H_opt_t = srt_to_matrix_torch(params)
        else:
            H_opt_t = torch.cat([params, params.new_tensor([1.0])]).view(3, 3)

    H_opt = torch_to_np(H_opt_t)
    H_opt = H_opt / H_opt[2, 2]

    return H_opt, H_init
