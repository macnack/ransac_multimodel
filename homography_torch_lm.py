"""
Pure-torch batched Levenberg-Marquardt for homography refinement.

Apples-to-apples competitor to `homography_theseus.py`:
  - same Mahalanobis-Huber residual + soft-barrier formulation,
  - same RANSAC initial guess (cv2),
  - batched (B, N, 2) inputs,
  - autograd-friendly (no .detach() on the path the user backprops through),
  - forward-mode Jacobian via torch.func.jacfwd — optimal when residual_dim
    (~N≈200) >> param_dim (4 or 8),
  - batched dense linear solve via torch.linalg.solve on (B, P, P) systems.

Why this exists separately from homography_torch.py: that file uses
torch.optim.LBFGS, which (a) doesn't batch, (b) lacks bound handling, (c) is
a quasi-Newton method that needs many function evals per accepted step. A
proper LM with analytical Hessian approximation J^T J is significantly
faster on small parameter counts and a fairer comparison to theseus.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.func import jacfwd, vmap

from .homography_theseus import (
    _chol_inv_cov,
    _full_h_to_matrix_batch,
    _huber_robust_residual,
    _params_init_from_H,
    _ransac_init,
    _srt_to_matrix_batch,
    _whitened_mahalanobis_resid,
)
from .parity_utils import np_to_torch, torch_to_np


def _err_fn_factory(
    model: str,
    f_scale: float,
    bounds_low: Optional[torch.Tensor],
    bounds_high: Optional[torch.Tensor],
    barrier_scale: float,
):
    """Returns err_fn(params, pts_A, means_B, L) -> residuals (R,) for one batch element."""

    def err_fn(params: torch.Tensor, pts_A: torch.Tensor, means_B: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # Operate on (1, *) inside so we can reuse the existing helpers.
        params_b = params.unsqueeze(0)
        pts_A_b = pts_A.unsqueeze(0)
        means_B_b = means_B.unsqueeze(0)
        L_b = L.unsqueeze(0)

        if model == "sRT":
            H = _srt_to_matrix_batch(params_b)
        else:
            H = _full_h_to_matrix_batch(params_b)

        maha = _whitened_mahalanobis_resid(H, pts_A_b, means_B_b, L_b)  # (1, N)
        robust = _huber_robust_residual(maha, f_scale).squeeze(0)        # (N,)

        if model == "sRT":
            over = barrier_scale * torch.nn.functional.relu(params - bounds_high)
            under = barrier_scale * torch.nn.functional.relu(bounds_low - params)
            return torch.nan_to_num(
                torch.cat([robust, over, under], dim=0), nan=0.0, posinf=1e6, neginf=-1e6
            )
        return torch.nan_to_num(robust, nan=0.0, posinf=1e6, neginf=-1e6)

    return err_fn


def _batched_lm(
    params0: torch.Tensor,           # (B, P)
    pts_A: torch.Tensor,             # (B, N, 2)
    means_B: torch.Tensor,           # (B, N, 2)
    L: torch.Tensor,                 # (B, N, 2, 2)
    err_fn,                          # single-element residual function
    max_iter: int,
    init_damping: float,
    damping_up: float,
    damping_down: float,
    abs_tol: float,
    rel_tol: float,
) -> torch.Tensor:
    """Vectorized batched LM. Returns refined params with grad attached."""
    B, P = params0.shape
    device, dtype = params0.device, params0.dtype
    eye_P = torch.eye(P, device=device, dtype=dtype)

    # vmap the per-element residual + jacobian across the batch dimension.
    # jacfwd is preferred over jacrev when the output dimension exceeds the
    # input dimension, which is overwhelmingly true here (R = N + 2P >> P).
    res_fn = vmap(err_fn)                            # (B,P), aux -> (B,R)
    jac_fn = vmap(jacfwd(err_fn, argnums=0))         # (B,P), aux -> (B,R,P)

    params = params0
    damping = torch.full((B,), init_damping, device=device, dtype=dtype)
    prev_cost = None
    abs_tol_t = torch.tensor(abs_tol, device=device, dtype=dtype)
    rel_tol_t = torch.tensor(rel_tol, device=device, dtype=dtype)

    for _ in range(max_iter):
        r = res_fn(params, pts_A, means_B, L)        # (B, R)
        J = jac_fn(params, pts_A, means_B, L)        # (B, R, P)

        cost = 0.5 * (r * r).sum(dim=-1)             # (B,)

        if prev_cost is not None:
            # Per-batch convergence: bool tensor stays on device. We sync
            # exactly once via .all().item() — the previous code synced
            # twice per iter via float(change) and float(ratio), and the
            # CUDA round-trip dominated runtime at small problem sizes.
            change = (prev_cost - cost).abs()
            ratio = change / (prev_cost.abs() + 1e-30)
            converged = (change < abs_tol_t) | (ratio < rel_tol_t)
            if bool(converged.all()):
                break
        prev_cost = cost

        JtJ = J.transpose(-1, -2) @ J                # (B, P, P)
        Jtr = (J.transpose(-1, -2) @ r.unsqueeze(-1)).squeeze(-1)  # (B, P)
        # Marquardt-style damping on the diagonal — scales with the diagonal
        # magnitude so it works across very different problem scalings.
        diag_boost = damping.unsqueeze(-1).unsqueeze(-1) * eye_P
        A = JtJ + diag_boost                         # (B, P, P)

        try:
            delta = torch.linalg.solve(A, Jtr.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            # Singular A on some batch element — bump damping for everyone and retry.
            damping = damping * damping_up
            continue

        new_params = params - delta                  # gradient-descent direction
        new_r = res_fn(new_params, pts_A, means_B, L)
        new_cost = 0.5 * (new_r * new_r).sum(dim=-1)

        # Per-batch-element step acceptance.
        accept = (new_cost < cost).unsqueeze(-1)     # (B, 1)
        params = torch.where(accept, new_params, params)
        damping = torch.where(accept.squeeze(-1), damping * damping_down, damping * damping_up)
        # Clamp damping so a long-rejected element doesn't explode out of float range.
        damping = damping.clamp(1e-12, 1e12)

    return params


def refine_homography_torch_lm_torch(
    pts_A: torch.Tensor,
    means_B: torch.Tensor,
    covs_B: torch.Tensor,
    H_init: torch.Tensor,
    model: str = "sRT",
    f_scale: float = 2.0,
    srt_bounds_low: tuple = (0.25, math.radians(-180.0), -60.0, -60.0),
    srt_bounds_high: tuple = (3.0, math.radians(180.0), 60.0, 60.0),
    barrier_k: float = 1.0,
    barrier_alpha: float = 1.0,
    max_iter: int = 100,
    init_damping: float = 1e-3,
    damping_up: float = 2.0,
    damping_down: float = 0.5,
    abs_err_tolerance: float = 1e-10,
    rel_err_tolerance: float = 1e-10,
) -> torch.Tensor:
    """
    Differentiable batched LM refinement. Inputs may be (N, ...) or (B, N, ...);
    a missing batch dim is added and removed transparently. H_init shape is
    (3, 3) or (B, 3, 3). Returns H_opt of shape (B, 3, 3).
    """
    if pts_A.dim() == 2:
        pts_A = pts_A.unsqueeze(0)
        means_B = means_B.unsqueeze(0)
        covs_B = covs_B.unsqueeze(0)
        H_init = H_init.unsqueeze(0)

    device, dtype = pts_A.device, pts_A.dtype

    # Whitening factor: chol(inv(cov + eps*I)).
    eye2 = torch.eye(2, dtype=dtype, device=device).expand_as(covs_B)
    covs_safe = covs_B + 1e-6 * eye2
    inv_covs = torch.linalg.inv(covs_safe)
    inv_covs = 0.5 * (inv_covs + inv_covs.transpose(-1, -2))
    L = torch.linalg.cholesky(inv_covs)              # (B, N, 2, 2)

    return _refine_from_L(
        pts_A, means_B, L, H_init,
        model=model, f_scale=f_scale,
        srt_bounds_low=srt_bounds_low, srt_bounds_high=srt_bounds_high,
        barrier_k=barrier_k, barrier_alpha=barrier_alpha,
        max_iter=max_iter, init_damping=init_damping,
        damping_up=damping_up, damping_down=damping_down,
        abs_err_tolerance=abs_err_tolerance, rel_err_tolerance=rel_err_tolerance,
    )


def _refine_from_L(
    pts_A: torch.Tensor,
    means_B: torch.Tensor,
    L: torch.Tensor,
    H_init: torch.Tensor,
    *,
    model: str,
    f_scale: float,
    srt_bounds_low: tuple,
    srt_bounds_high: tuple,
    barrier_k: float,
    barrier_alpha: float,
    max_iter: int,
    init_damping: float,
    damping_up: float,
    damping_down: float,
    abs_err_tolerance: float,
    rel_err_tolerance: float,
) -> torch.Tensor:
    """Internal: run the LM given a precomputed Cholesky factor L of inv(cov).

    Lets ``optimize_homography_torch_lm`` reuse the L it already built from
    numpy without round-tripping through ``cov`` and re-Cholesky'ing.
    """
    B, N, _ = pts_A.shape
    device, dtype = pts_A.device, pts_A.dtype

    H0 = H_init / H_init[..., 2:3, 2:3]
    if model == "sRT":
        a = H0[..., 0, 0]
        b = H0[..., 1, 0]
        s = torch.sqrt(a * a + b * b)
        theta = torch.atan2(b, a)
        params0 = torch.stack([s, theta, H0[..., 0, 2], H0[..., 1, 2]], dim=-1)
    else:
        params0 = H0.reshape(B, 9)[:, :8]

    bounds_low = torch.as_tensor(srt_bounds_low, dtype=dtype, device=device) if model == "sRT" else None
    bounds_high = torch.as_tensor(srt_bounds_high, dtype=dtype, device=device) if model == "sRT" else None
    barrier_scale = math.sqrt(max(barrier_alpha * barrier_k, 0.0))

    err_fn = _err_fn_factory(model, f_scale, bounds_low, bounds_high, barrier_scale)

    params_opt = _batched_lm(
        params0, pts_A, means_B, L, err_fn,
        max_iter=max_iter, init_damping=init_damping,
        damping_up=damping_up, damping_down=damping_down,
        abs_tol=abs_err_tolerance, rel_tol=rel_err_tolerance,
    )

    if model == "sRT":
        H_opt = _srt_to_matrix_batch(params_opt)
    else:
        H_opt = _full_h_to_matrix_batch(params_opt)
    return H_opt / H_opt[..., 2:3, 2:3]


def optimize_homography_torch_lm(
    pts_A,
    means_B,
    covs_B,
    peaks_B=None,
    model: str = "full",
    use_means_for_ransac: bool = False,
    quiet: bool = False,
    ransac_method: int = cv2.USAC_FAST,
    ransac_reproj_threshold: float = 3.0,
    ransac_max_iters: int = 5000,
    ransac_confidence: float = 0.995,
    f_scale: float = 2.0,
    srt_bounds_low: tuple = (0.25, math.radians(-180.0), -60.0, -60.0),
    srt_bounds_high: tuple = (3.0, math.radians(180.0), 60.0, 60.0),
    barrier_k: float = 1.0,
    barrier_alpha: float = 1.0,
    max_iter: int = 100,
    init_damping: float = 1e-3,
    damping_up: float = 2.0,
    damping_down: float = 0.5,
    abs_err_tolerance: float = 1e-10,
    rel_err_tolerance: float = 1e-10,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop-in replacement for `optimize_homography` powered by a hand-rolled
    batched torch LM. Returns (H_opt_np, H_init_np) like the original.
    """
    pts_A = np.asarray(pts_A)
    means_B = np.asarray(means_B)
    covs_B = np.asarray(covs_B)

    N = pts_A.shape[0]
    if N < 4:
        raise ValueError("At least 4 points are required to compute a homography.")

    H_init_norm, _inlier_mask = _ransac_init(
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

    L_np = _chol_inv_cov(covs_B.astype(np.float64))

    pts_A_t = np_to_torch(pts_A.astype(np.float64), device=device, dtype=dtype).reshape(1, N, 2)
    means_B_t = np_to_torch(means_B.astype(np.float64), device=device, dtype=dtype).reshape(1, N, 2)
    L_t = np_to_torch(L_np, device=device, dtype=dtype).reshape(1, N, 2, 2)
    H_init_t = np_to_torch(H_init_norm, device=device, dtype=dtype).reshape(1, 3, 3)

    with torch.no_grad():
        H_opt_t = _refine_from_L(
            pts_A_t, means_B_t, L_t, H_init_t,
            model=model, f_scale=f_scale,
            srt_bounds_low=srt_bounds_low, srt_bounds_high=srt_bounds_high,
            barrier_k=barrier_k, barrier_alpha=barrier_alpha,
            max_iter=max_iter, init_damping=init_damping,
            damping_up=damping_up, damping_down=damping_down,
            abs_err_tolerance=abs_err_tolerance, rel_err_tolerance=rel_err_tolerance,
        )[0]

    H_opt = torch_to_np(H_opt_t)
    H_opt = H_opt / H_opt[2, 2]
    return H_opt, H_init_norm
