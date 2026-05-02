"""
Differentiable replacement for `optimize_homography` (scipy.optimize.least_squares)
built on top of Theseus.

Pipeline mirrors `homography.py` / `homography_torch.py`:
  - cv2.findHomography RANSAC -> initial guess
  - per-correspondence Mahalanobis residual whitened by chol(inv_cov)
  - Huber robust loss (matches scipy's loss="huber", f_scale=...)
  - inner-loop LM via theseus.LevenbergMarquardt wrapped in TheseusLayer

Two entrypoints:
  - optimize_homography_theseus(pts_A_np, ...)  -> (H_opt_np, H_init_np)
        drop-in replacement for optimize_homography
  - refine_homography_theseus_torch(H_init_t, pts_A_t, ...) -> H_opt_t
        keeps the autograd graph so callers can backprop a downstream
        geometry loss through the refinement.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# The vendored theseus / torchlie / torchkin checkouts live under
#   <repo>/theseus/{theseus,torchlie,torchkin}/
# Each top-level dir lacks __init__.py, so when CWD == repo root Python
# discovers them as namespace packages and shadows the editable installs.
# Inserting the inner parent dirs at the front of sys.path makes the real
# packages (with __init__.py) win.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_THESEUS_DIR = _os.path.join(_REPO_ROOT, "theseus")
for _p in (
    _os.path.join(_THESEUS_DIR, "torchkin"),
    _os.path.join(_THESEUS_DIR, "torchlie"),
    _THESEUS_DIR,
):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

import math
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

import theseus as th  # noqa: E402

from .parity_utils import np_to_torch, torch_to_np


# ---------------------------------------------------------------------------
# helpers (torch-side)
# ---------------------------------------------------------------------------


def _project_points_torch(H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """H: (B,3,3); pts: (B,N,2). Returns (B,N,2)."""
    ones = pts.new_ones(pts.shape[:-1] + (1,))
    pts_h = torch.cat([pts, ones], dim=-1)  # (B,N,3)
    proj = torch.einsum("bij,bnj->bni", H, pts_h)
    z = proj[..., 2:3] + 1e-6
    return proj[..., :2] / z


def _srt_to_matrix_batch(params: torch.Tensor) -> torch.Tensor:
    """params: (B,4) [s, theta, tx, ty]. Returns H (B,3,3)."""
    s = torch.clamp(params[:, 0], 0.0, 4.0)
    theta = torch.clamp(params[:, 1], math.radians(-90.0), math.radians(90.0))
    tx = params[:, 2]
    ty = params[:, 3]
    c, si = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(s)
    one = torch.ones_like(s)
    row0 = torch.stack([s * c, -s * si, tx], dim=-1)
    row1 = torch.stack([s * si, s * c, ty], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)  # (B,3,3)


def _full_h_to_matrix_batch(params: torch.Tensor) -> torch.Tensor:
    """params: (B,8) -> H (B,3,3) with H[2,2] forced to 1."""
    ones = params.new_ones(params.shape[0], 1)
    full = torch.cat([params, ones], dim=-1)
    return full.reshape(-1, 3, 3)


def _whitened_mahalanobis_resid(
    H: torch.Tensor,
    pts_A: torch.Tensor,
    means_B: torch.Tensor,
    L_invcov: torch.Tensor,
) -> torch.Tensor:
    """
    Returns per-correspondence Mahalanobis distance: (B, N).
    L_invcov is the lower-triangular Cholesky factor of inv_cov, so
    |L^T (proj - mean)|^2 == (proj-mean)^T inv_cov (proj-mean).
    """
    proj = _project_points_torch(H, pts_A)
    err = proj - means_B  # (B,N,2)
    whitened = torch.einsum("bnji,bnj->bni", L_invcov, err)  # apply L^T
    maha_sq = (whitened ** 2).sum(dim=-1)
    return torch.sqrt(torch.clamp(maha_sq, min=1e-12))


def _huber_robust_residual(maha: torch.Tensor, f_scale: float) -> torch.Tensor:
    """
    Convert per-correspondence Mahalanobis distance to scipy's Huber-robust
    pseudo-residual so that 0.5 * sum(robust^2) == 0.5 * sum(huber(maha^2)).

    Matches scipy.optimize.least_squares(loss="huber", f_scale=s):
        z = (maha / s)^2
        rho(z) = z         if z <= 1
                 2*sqrt(z) - 1 otherwise
        robust_residual = s * sqrt(rho(z))
    """
    s = float(f_scale)
    z = (maha / s) ** 2
    rho = torch.where(z <= 1.0, z, 2.0 * torch.sqrt(z + 1e-12) - 1.0)
    return s * torch.sqrt(rho + 1e-12)


# ---------------------------------------------------------------------------
# objective builder
# ---------------------------------------------------------------------------


def _ransac_init(
    pts_A: np.ndarray,
    means_B: np.ndarray,
    peaks_B: Optional[np.ndarray],
    use_means_for_ransac: bool,
    ransac_method: int,
    ransac_reproj_threshold: float,
    ransac_max_iters: int,
    ransac_confidence: float,
    quiet: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if use_means_for_ransac:
        ransac_pts_B = means_B
    else:
        ransac_pts_B = peaks_B if peaks_B is not None else means_B
    H_init, mask = cv2.findHomography(
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
        H_init = np.eye(3, dtype=np.float64)
        mask = np.zeros((pts_A.shape[0], 1), dtype=np.uint8)
    return H_init / H_init[2, 2], mask


def _params_init_from_H(H_init_norm: np.ndarray, model: str) -> np.ndarray:
    if model == "sRT":
        a, b = H_init_norm[0, 0], H_init_norm[1, 0]
        s = float(np.sqrt(a * a + b * b))
        theta = float(np.arctan2(b, a))
        return np.array([s, theta, H_init_norm[0, 2], H_init_norm[1, 2]], dtype=np.float64)
    return H_init_norm.flatten()[:8].astype(np.float64)


def _build_layer(
    n_points: int,
    model: str,
    f_scale: float,
    srt_scale_reg_weight: float,
    srt_rot_reg_weight: float,
    srt_bounds_low: tuple,
    srt_bounds_high: tuple,
    barrier_k: float,
    barrier_alpha: float,
    max_iter: int,
    step_size: float,
    damping: float,
    damping_update_factor: float,
    abs_err_tolerance: float,
    rel_err_tolerance: float,
    dtype: torch.dtype,
):
    """Return (theseus_layer, optim_var_name, residual_dim)."""
    param_dim = 4 if model == "sRT" else 8
    # One Mahalanobis-Huber residual per correspondence, plus 2*param_dim
    # one-sided barrier residuals (over_i and under_i for each sRT variable).
    # Each barrier residual is exactly 0 inside the feasible set, so it does
    # not perturb the inner-loop fit when bounds are not active.
    n_barrier = 2 * param_dim if model == "sRT" else 0
    residual_dim = n_points + n_barrier

    bounds_low = torch.tensor(srt_bounds_low, dtype=dtype) if model == "sRT" else None
    bounds_high = torch.tensor(srt_bounds_high, dtype=dtype) if model == "sRT" else None

    def err_fn(optim_vars, aux_vars):
        params = optim_vars[0].tensor  # (B, param_dim)
        pts_A = aux_vars[0].tensor      # (B, N, 2)
        means_B = aux_vars[1].tensor    # (B, N, 2)
        L = aux_vars[2].tensor          # (B, N, 2, 2) chol(inv_cov)

        if model == "sRT":
            H = _srt_to_matrix_batch(params)
        else:
            H = _full_h_to_matrix_batch(params)

        maha = _whitened_mahalanobis_resid(H, pts_A, means_B, L)  # (B, N)
        robust = _huber_robust_residual(maha, f_scale)            # (B, N)

        if model == "sRT":
            lo = bounds_low.to(device=params.device, dtype=params.dtype)
            hi = bounds_high.to(device=params.device, dtype=params.dtype)
            # Quadratic-in-cost one-sided penalty: residual = sqrt(α*k) * relu(.)
            # When fed to LM (which minimizes 0.5 * ||r||²) this contributes
            #   0.5 * α * k * relu(over)²
            # exactly 0 inside the feasible region, ramping up smoothly outside.
            scale = math.sqrt(max(barrier_alpha * barrier_k, 0.0))
            over = scale * torch.nn.functional.relu(params - hi)   # (B, param_dim)
            under = scale * torch.nn.functional.relu(lo - params)  # (B, param_dim)
            barrier = torch.cat([over, under], dim=-1)             # (B, 2*param_dim)
            out = torch.cat([robust, barrier], dim=-1)             # (B, N + 2*param_dim)
        else:
            out = robust

        # Defensive: any NaN/Inf from sqrt of degenerate covariance or from a
        # singular projection would silently produce a singular H downstream.
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    optim_var_name = "params"
    init_zeros = torch.zeros(1, param_dim, dtype=dtype)
    pts_zero = torch.zeros(1, n_points, 2, dtype=dtype)
    L_zero = torch.zeros(1, n_points, 2, 2, dtype=dtype)

    params_var = th.Vector(tensor=init_zeros, name=optim_var_name)
    pts_A_var = th.Variable(tensor=pts_zero, name="pts_A")
    means_B_var = th.Variable(tensor=pts_zero.clone(), name="means_B")
    L_var = th.Variable(tensor=L_zero, name="L_invcov")

    # Default ScaleCostWeight(1.0) creates a float32 tensor and clashes with
    # our float64 optim/aux vars; bind the weight to the requested dtype.
    cost_weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=dtype))

    cost = th.AutoDiffCostFunction(
        optim_vars=[params_var],
        err_fn=err_fn,
        dim=residual_dim,
        aux_vars=[pts_A_var, means_B_var, L_var],
        cost_weight=cost_weight,
        # "vmap" vectorizes Jacobian computation across the residual rows;
        # "dense" loops, which is O(N) slower for our N≈200 residuals.
        autograd_mode="vmap",
    )

    objective = th.Objective(dtype=dtype)
    objective.add(cost)

    # Identity-pulling regularization for sRT is expressed as a separate
    # th.Difference cost (the idiomatic theseus pattern) rather than appended
    # to the per-correspondence residual. ScaleCostWeight is sqrt of the L2
    # weight because LM minimizes 0.5 * ||residual||^2.
    if model == "sRT":
        identity_sRT = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
        # Per-component weighting: scale and rotation use different reg weights;
        # tx/ty are not regularized (scipy version doesn't either).
        diag = torch.tensor(
            [
                math.sqrt(max(srt_scale_reg_weight, 0.0)),
                math.sqrt(max(srt_rot_reg_weight, 0.0)) * (180.0 / math.pi),
                0.0,
                0.0,
            ],
            dtype=dtype,
        ).reshape(1, -1)
        if float(diag.abs().sum()) > 0.0:
            reg_weight = th.DiagonalCostWeight(diag)
            id_var = th.Vector(tensor=identity_sRT, name="identity_sRT")
            reg_cf = th.Difference(
                params_var, target=id_var, cost_weight=reg_weight, name="reg_sRT"
            )
            objective.add(reg_cf)

    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=max_iter,
        step_size=step_size,
        abs_err_tolerance=abs_err_tolerance,
        rel_err_tolerance=rel_err_tolerance,
    )
    # damping / damping_update_factor are set on the instance because the LM
    # constructor's keyword set varies across theseus versions.
    for attr, val in (("damping", damping), ("damping_update_factor", damping_update_factor)):
        if hasattr(optimizer, attr):
            setattr(optimizer, attr, val)
    layer = th.TheseusLayer(optimizer)
    return layer, optim_var_name, residual_dim


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def _chol_inv_cov(covs_np: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor of inv(cov + eps*I) per correspondence."""
    covs_safe = covs_np + np.eye(2, dtype=covs_np.dtype) * 1e-6
    inv_covs = np.linalg.inv(covs_safe)
    # Symmetrize to guard against numerical drift before Cholesky.
    inv_covs = 0.5 * (inv_covs + inv_covs.swapaxes(-1, -2))
    return np.linalg.cholesky(inv_covs)


def optimize_homography_theseus(
    pts_A,
    means_B,
    covs_B,
    peaks_B=None,
    model: str = "full",
    use_means_for_ransac: bool = False,
    verbose: int = 0,
    quiet: bool = False,
    ransac_method: int = cv2.RANSAC,
    ransac_reproj_threshold: float = 3.0,
    ransac_max_iters: int = 5000,
    ransac_confidence: float = 0.995,
    f_scale: float = 2.0,
    srt_scale_reg_weight: float = 0.01,
    srt_rot_reg_weight: float = 0.001,
    srt_bounds_low: tuple = (0.25, math.radians(-180.0), -60.0, -60.0),
    srt_bounds_high: tuple = (3.0, math.radians(180.0), 60.0, 60.0),
    barrier_k: float = 1.0,
    barrier_alpha: float = 1.0,
    max_iter: int = 100,
    step_size: float = 0.1,
    damping: float = 1e-6,
    damping_update_factor: float = 1.5,
    abs_err_tolerance: float = 1e-10,
    rel_err_tolerance: float = 1e-10,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    return_details: bool = False,
):
    """
    Drop-in replacement for `optimize_homography` powered by Theseus's
    Levenberg-Marquardt. Returns (H_opt_np, H_init_np) like the original.
    """
    pts_A = np.asarray(pts_A)
    means_B = np.asarray(means_B)
    covs_B = np.asarray(covs_B)

    N = pts_A.shape[0]
    if N < 4:
        raise ValueError("At least 4 points are required to compute a homography.")

    H_init_norm, inlier_mask = _ransac_init(
        pts_A,
        means_B,
        peaks_B if peaks_B is None else np.asarray(peaks_B),
        use_means_for_ransac,
        ransac_method,
        ransac_reproj_threshold,
        ransac_max_iters,
        ransac_confidence,
        quiet,
    )

    L = _chol_inv_cov(covs_B.astype(np.float64))

    layer, optim_var_name, _ = _build_layer(
        n_points=N,
        model=model,
        f_scale=f_scale,
        srt_scale_reg_weight=srt_scale_reg_weight,
        srt_rot_reg_weight=srt_rot_reg_weight,
        srt_bounds_low=srt_bounds_low,
        srt_bounds_high=srt_bounds_high,
        barrier_k=barrier_k,
        barrier_alpha=barrier_alpha,
        max_iter=max_iter,
        step_size=step_size,
        damping=damping,
        damping_update_factor=damping_update_factor,
        abs_err_tolerance=abs_err_tolerance,
        rel_err_tolerance=rel_err_tolerance,
        dtype=dtype,
    )
    layer = layer.to(device)

    init_params = _params_init_from_H(H_init_norm, model)

    inputs = {
        optim_var_name: np_to_torch(init_params, device=device, dtype=dtype).reshape(1, -1),
        "pts_A": np_to_torch(pts_A.astype(np.float64), device=device, dtype=dtype).reshape(1, N, 2),
        "means_B": np_to_torch(means_B.astype(np.float64), device=device, dtype=dtype).reshape(1, N, 2),
        "L_invcov": np_to_torch(L, device=device, dtype=dtype).reshape(1, N, 2, 2),
    }

    with torch.no_grad():
        _, info = layer.forward(
            inputs,
            optimizer_kwargs={
                "verbose": bool(verbose),
                "track_err_history": False,
                "backward_mode": "implicit",
            },
        )

    params_opt = layer.objective.get_optim_var(optim_var_name).tensor  # (1, param_dim)
    if model == "sRT":
        H_opt_t = _srt_to_matrix_batch(params_opt)[0]
    else:
        H_opt_t = _full_h_to_matrix_batch(params_opt)[0]
    H_opt = torch_to_np(H_opt_t)
    H_opt = H_opt / H_opt[2, 2]

    if return_details:
        details = {
            "num_correspondences": int(N),
            "num_inliers": int(np.sum(inlier_mask)) if inlier_mask is not None else 0,
            "inlier_ratio": float(np.mean(inlier_mask)) if inlier_mask is not None else 0.0,
            "optimization_status": int(getattr(info, "status", [0])[0])
            if hasattr(info, "status") else 0,
            "best_err": float(getattr(info, "best_err", torch.tensor(float("nan"))).item())
            if hasattr(info, "best_err") else float("nan"),
            "converged_iter": int(getattr(info, "converged_iter", torch.tensor(-1)).item())
            if hasattr(info, "converged_iter") else -1,
        }
        return H_opt, H_init_norm, details

    return H_opt, H_init_norm


def refine_homography_theseus_torch(
    pts_A: torch.Tensor,
    means_B: torch.Tensor,
    covs_B: torch.Tensor,
    H_init: torch.Tensor,
    model: str = "sRT",
    f_scale: float = 2.0,
    srt_scale_reg_weight: float = 0.01,
    srt_rot_reg_weight: float = 0.001,
    srt_bounds_low: tuple = (0.25, math.radians(-180.0), -60.0, -60.0),
    srt_bounds_high: tuple = (3.0, math.radians(180.0), 60.0, 60.0),
    barrier_k: float = 1.0,
    barrier_alpha: float = 1.0,
    max_iter: int = 100,
    step_size: float = 0.1,
    damping: float = 1e-6,
    damping_update_factor: float = 1.5,
    abs_err_tolerance: float = 1e-10,
    rel_err_tolerance: float = 1e-10,
    backward_mode: str = "implicit",
    verbose: bool = False,
) -> torch.Tensor:
    """
    Differentiable refinement. All inputs are torch tensors; the returned
    H_opt keeps the autograd graph so callers can backprop a downstream
    geometry loss through the Theseus inner-loop optimizer.

    Shapes (B = batch):
      pts_A:   (B, N, 2)
      means_B: (B, N, 2)
      covs_B:  (B, N, 2, 2)
      H_init:  (B, 3, 3)   — used to seed the optim variable

    Returns H_opt: (B, 3, 3).
    """
    if pts_A.dim() == 2:
        pts_A = pts_A.unsqueeze(0)
        means_B = means_B.unsqueeze(0)
        covs_B = covs_B.unsqueeze(0)
        H_init = H_init.unsqueeze(0)

    B, N, _ = pts_A.shape
    # Theseus's AutoDiffCostFunction handles a leading batch dim natively as
    # long as N is fixed across the batch (cost dim depends on N, not B).
    # Callers with varying N must loop — pad+mask is also possible but adds
    # bias on the masked rows.

    device = pts_A.device
    dtype = pts_A.dtype

    # Whitening factor: chol(inv(cov + eps*I)).
    eye2 = torch.eye(2, dtype=dtype, device=device).expand_as(covs_B)
    covs_safe = covs_B + 1e-6 * eye2
    inv_covs = torch.linalg.inv(covs_safe)
    inv_covs = 0.5 * (inv_covs + inv_covs.transpose(-1, -2))
    L = torch.linalg.cholesky(inv_covs)  # (B, N, 2, 2)

    layer, optim_var_name, _ = _build_layer(
        n_points=N,
        model=model,
        f_scale=f_scale,
        srt_scale_reg_weight=srt_scale_reg_weight,
        srt_rot_reg_weight=srt_rot_reg_weight,
        srt_bounds_low=srt_bounds_low,
        srt_bounds_high=srt_bounds_high,
        barrier_k=barrier_k,
        barrier_alpha=barrier_alpha,
        max_iter=max_iter,
        step_size=step_size,
        damping=damping,
        damping_update_factor=damping_update_factor,
        abs_err_tolerance=abs_err_tolerance,
        rel_err_tolerance=rel_err_tolerance,
        dtype=dtype,
    )
    layer = layer.to(device)

    # Seed optim variable from H_init.
    H0 = (H_init / H_init[..., 2:3, 2:3]).detach()
    if model == "sRT":
        a = H0[..., 0, 0]
        b = H0[..., 1, 0]
        s = torch.sqrt(a * a + b * b)
        theta = torch.atan2(b, a)
        params0 = torch.stack([s, theta, H0[..., 0, 2], H0[..., 1, 2]], dim=-1)
    else:
        params0 = H0.reshape(B, 9)[:, :8]

    inputs = {
        optim_var_name: params0,
        "pts_A": pts_A,
        "means_B": means_B,
        "L_invcov": L,
    }

    _, _info = layer.forward(
        inputs,
        optimizer_kwargs={
            "verbose": bool(verbose),
            "track_err_history": False,
            "backward_mode": backward_mode,
        },
    )

    params_opt = layer.objective.get_optim_var(optim_var_name).tensor  # (B, param_dim)
    if model == "sRT":
        H_opt = _srt_to_matrix_batch(params_opt)
    else:
        H_opt = _full_h_to_matrix_batch(params_opt)
    H_opt = H_opt / H_opt[..., 2:3, 2:3]
    return H_opt
