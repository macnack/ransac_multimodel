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

from dataclasses import dataclass, field

from .homography_theseus import (
    _chol_inv_cov,
    _full_h_to_matrix_batch,
    _huber_robust_residual,
    _params_init_from_H,
    _srt_to_matrix_batch,
    _whitened_mahalanobis_resid,
)
from .parity_utils import np_to_torch, torch_to_np
from .ransac_init import ransac_init


@dataclass
class LMHistory:
    """Per-iteration trajectory of the batched LM solve.

    All tensors have shape ``(n_iters, B)`` (or ``(n_iters,)`` for scalars).
    ``n_iters`` is the actual number of iterations run, which can be less
    than ``max_iter`` when the convergence check fires early.

    Use cases:
    - Plot per-batch-element ``cost`` vs iter to debug non-convergence.
    - Inspect ``accept`` to see which iterations were rejected per element.
    - Look at ``damping`` to see whether LM was in trust-region (damping
      growing) vs Gauss-Newton (damping shrinking) regime.
    - Stream to neptune/tensorboard via the ``logger`` callback (see
      :func:`refine_homography_torch_lm_torch`).
    """
    cost: torch.Tensor                       # (n_iters, B) — 0.5 * ||r||²
    damping: torch.Tensor                    # (n_iters, B) — Marquardt λ
    accept: torch.Tensor                     # (n_iters, B) bool — step accepted?
    n_iters: int                             # actual iterations run
    converged: torch.Tensor                  # (B,) bool — final per-batch convergence flag
    final_cost: torch.Tensor                 # (B,) — convenience, == cost[-1]

    def to_numpy(self) -> "NumpyLMHistory":
        """Convert to NumpyLMHistory for use with divergence guard (pure numpy).

        Detaches from autograd and moves to CPU as needed. Zero-copy for tensors
        already on CPU; ~B*40 bytes for typical batches.
        """
        from .divergence_guard import NumpyLMHistory

        return NumpyLMHistory(
            cost_init=self.cost[0].detach().cpu().numpy().astype(np.float64),
            cost_final=self.final_cost.detach().cpu().numpy().astype(np.float64),
            n_iters=self.n_iters,
            converged=self.converged.detach().cpu().numpy().astype(bool),
        )


def lm_history_to_numpy(history: LMHistory) -> "NumpyLMHistory":
    """Convert torch LMHistory to numpy NumpyLMHistory for divergence guard.

    Standalone version of LMHistory.to_numpy(); useful when you have the
    history but not access to call the method directly.

    Parameters
    ----------
    history
        LMHistory from refine_homography_torch_lm_torch.

    Returns
    -------
    NumpyLMHistory
        Pure numpy, ready for apply_divergence_guard.
    """
    return history.to_numpy()


def _err_fn_factory(
    model: str,
    f_scale: float,
    bounds_low: Optional[torch.Tensor],
    bounds_high: Optional[torch.Tensor],
    barrier_scale: float,
):
    """Returns err_fn(params, pts_A, means_B, L, mask) -> residuals (R,).

    ``mask`` is a (N,) float tensor per batch element with 1 for real
    correspondences and 0 for padded slots. We multiply the per-corr
    Mahalanobis-Huber residual by mask so padded slots contribute 0 to
    the cost AND their Jacobian rows are 0 (they don't influence the
    LM normal equations). Pass an all-ones mask when no padding is needed.
    """

    def err_fn(
        params: torch.Tensor,
        pts_A: torch.Tensor,
        means_B: torch.Tensor,
        L: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
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
        # Zero out padded slots; padded means_B / L are arbitrary garbage.
        robust = robust * mask

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
    mask: torch.Tensor,              # (B, N) float — 1 real, 0 padded
    err_fn,                          # single-element residual function
    max_iter: int,
    init_damping: float,
    damping_up: float,
    damping_down: float,
    abs_tol: float,
    rel_tol: float,
    track_history: bool = False,
    logger=None,                     # optional Callable(iter, cost, damping, accept)
):
    """Vectorized batched LM. Returns refined params with grad attached.

    If ``track_history`` is True, additionally returns (params, history)
    where history is a :class:`LMHistory`. ``logger`` is an optional
    callback called once per iteration with
    ``(iter_idx: int, cost: (B,), damping: (B,), accept: (B,) bool)``.
    Useful for streaming to neptune / tensorboard without paying the cost
    of materializing the full history tensor.
    """
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

    cost_hist: list = []
    damping_hist: list = []
    accept_hist: list = []
    converged_final = torch.zeros((B,), dtype=torch.bool, device=device)
    n_iters = 0

    for it in range(max_iter):
        r = res_fn(params, pts_A, means_B, L, mask)  # (B, R)
        J = jac_fn(params, pts_A, means_B, L, mask)  # (B, R, P)

        cost = 0.5 * (r * r).sum(dim=-1)             # (B,)

        if prev_cost is not None:
            # Per-batch convergence: bool tensor stays on device. We sync
            # exactly once via .all().item() — the previous code synced
            # twice per iter via float(change) and float(ratio), and the
            # CUDA round-trip dominated runtime at small problem sizes.
            change = (prev_cost - cost).abs()
            ratio = change / (prev_cost.abs() + 1e-30)
            converged_final = (change < abs_tol_t) | (ratio < rel_tol_t)
            if bool(converged_final.all()):
                if track_history:
                    # Final cost row, no step taken — log a no-op accept row
                    # for shape consistency.
                    cost_hist.append(cost.detach())
                    damping_hist.append(damping.detach())
                    accept_hist.append(torch.zeros((B,), dtype=torch.bool, device=device))
                if logger is not None:
                    logger(it, cost.detach(), damping.detach(),
                           torch.zeros((B,), dtype=torch.bool, device=device))
                n_iters = it + 1
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
            if track_history:
                cost_hist.append(cost.detach())
                damping_hist.append(damping.detach())
                accept_hist.append(torch.zeros((B,), dtype=torch.bool, device=device))
            if logger is not None:
                logger(it, cost.detach(), damping.detach(),
                       torch.zeros((B,), dtype=torch.bool, device=device))
            n_iters = it + 1
            continue

        new_params = params - delta                  # gradient-descent direction
        new_r = res_fn(new_params, pts_A, means_B, L, mask)
        new_cost = 0.5 * (new_r * new_r).sum(dim=-1)

        # Per-batch-element step acceptance.
        accept = (new_cost < cost).unsqueeze(-1)     # (B, 1)
        params = torch.where(accept, new_params, params)
        damping = torch.where(accept.squeeze(-1), damping * damping_down, damping * damping_up)
        # Clamp damping so a long-rejected element doesn't explode out of float range.
        damping = damping.clamp(1e-12, 1e12)

        if track_history:
            cost_hist.append(cost.detach())
            damping_hist.append(damping.detach())
            accept_hist.append(accept.squeeze(-1).detach())
        if logger is not None:
            logger(it, cost.detach(), damping.detach(), accept.squeeze(-1).detach())
        n_iters = it + 1

    if not track_history:
        return params

    if not cost_hist:  # max_iter == 0 edge case
        empty_b = torch.zeros((0, B), dtype=dtype, device=device)
        empty_b_bool = torch.zeros((0, B), dtype=torch.bool, device=device)
        history = LMHistory(
            cost=empty_b, damping=empty_b.clone(), accept=empty_b_bool,
            n_iters=0, converged=converged_final,
            final_cost=torch.zeros((B,), dtype=dtype, device=device),
        )
    else:
        cost_t = torch.stack(cost_hist, dim=0)
        damping_t = torch.stack(damping_hist, dim=0)
        accept_t = torch.stack(accept_hist, dim=0)
        history = LMHistory(
            cost=cost_t, damping=damping_t, accept=accept_t,
            n_iters=n_iters, converged=converged_final,
            final_cost=cost_t[-1],
        )
    return params, history


def refine_homography_torch_lm_torch(
    pts_A: torch.Tensor,
    means_B: torch.Tensor,
    covs_B: torch.Tensor,
    H_init: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
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
    track_history: bool = False,
    logger=None,
):
    """
    Differentiable batched LM refinement. Inputs may be (N, ...) or (B, N, ...);
    a missing batch dim is added and removed transparently. H_init shape is
    (3, 3) or (B, 3, 3). Returns H_opt of shape (B, 3, 3).

    ``mask`` (optional, shape (B, N) or (N,)) marks real correspondences (1)
    vs padded slots (0). Padded slots contribute 0 to the cost AND to the
    Jacobian — equivalent to running the LM on the un-padded data, but lets
    you batch frames that have different actual N by padding to N_max.
    Pass ``None`` (default) when no padding is involved.
    """
    if pts_A.dim() == 2:
        pts_A = pts_A.unsqueeze(0)
        means_B = means_B.unsqueeze(0)
        covs_B = covs_B.unsqueeze(0)
        H_init = H_init.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    device, dtype = pts_A.device, pts_A.dtype

    # Whitening factor: chol(inv(cov + eps*I)).
    # For padded slots, covs_B may be zero or arbitrary — Cholesky needs SPD,
    # so we replace padded covariances with identity before inverting.
    if mask is not None:
        # mask: (B, N), broadcast to (B, N, 2, 2).
        mask_4d = mask.bool()[..., None, None]
        eye2_full = torch.eye(2, dtype=dtype, device=device).expand_as(covs_B)
        covs_B = torch.where(mask_4d, covs_B, eye2_full)

    eye2 = torch.eye(2, dtype=dtype, device=device).expand_as(covs_B)
    covs_safe = covs_B + 1e-6 * eye2
    inv_covs = torch.linalg.inv(covs_safe)
    inv_covs = 0.5 * (inv_covs + inv_covs.transpose(-1, -2))
    L = torch.linalg.cholesky(inv_covs)              # (B, N, 2, 2)

    return _refine_from_L(
        pts_A, means_B, L, H_init, mask=mask,
        model=model, f_scale=f_scale,
        srt_bounds_low=srt_bounds_low, srt_bounds_high=srt_bounds_high,
        barrier_k=barrier_k, barrier_alpha=barrier_alpha,
        max_iter=max_iter, init_damping=init_damping,
        damping_up=damping_up, damping_down=damping_down,
        abs_err_tolerance=abs_err_tolerance, rel_err_tolerance=rel_err_tolerance,
        track_history=track_history, logger=logger,
    )


def _refine_from_L(
    pts_A: torch.Tensor,
    means_B: torch.Tensor,
    L: torch.Tensor,
    H_init: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
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
    track_history: bool = False,
    logger=None,
):
    """Internal: run the LM given a precomputed Cholesky factor L of inv(cov).

    Lets ``optimize_homography_torch_lm`` reuse the L it already built from
    numpy without round-tripping through ``cov`` and re-Cholesky'ing.
    ``mask`` is broadcast to (B, N); pass None to use all-ones (no padding).
    """
    B, N, _ = pts_A.shape
    device, dtype = pts_A.device, pts_A.dtype
    if mask is None:
        mask = torch.ones((B, N), dtype=dtype, device=device)
    else:
        mask = mask.to(dtype=dtype, device=device)

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

    lm_out = _batched_lm(
        params0, pts_A, means_B, L, mask, err_fn,
        max_iter=max_iter, init_damping=init_damping,
        damping_up=damping_up, damping_down=damping_down,
        abs_tol=abs_err_tolerance, rel_tol=rel_err_tolerance,
        track_history=track_history, logger=logger,
    )
    if track_history:
        params_opt, history = lm_out
    else:
        params_opt = lm_out
        history = None

    if model == "sRT":
        H_opt = _srt_to_matrix_batch(params_opt)
    else:
        H_opt = _full_h_to_matrix_batch(params_opt)
    H_opt = H_opt / H_opt[..., 2:3, 2:3]
    if track_history:
        return H_opt, history
    return H_opt


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


# --------------------------------------------------------------------------- #
# Padding helper for heterogeneous batched LM                                 #
# --------------------------------------------------------------------------- #


def pad_for_batched_lm(
    per_frame: list,
    H_init_per_frame: list,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Pack per-frame (pts_A, means_B, peaks_B, covs_B) into padded tensors.

    Lets ``refine_homography_torch_lm_torch`` run on a batch where each frame
    has a different number of correspondences. Frames with fewer than N_max
    correspondences get zero-padded slots; ``mask`` marks them so the LM
    cost ignores them.

    Parameters
    ----------
    per_frame : list of (pts_A, means_B, peaks_B, covs_B) numpy tuples
        As returned by ``find_gaussians_torch_batch`` or per-frame extraction.
    H_init_per_frame : list of (3, 3) numpy arrays
        RANSAC init for each frame. Frames with N < 4 should still have a
        valid 3x3 (e.g. identity).
    device, dtype
        Where to place the returned tensors.

    Returns
    -------
    (pts_A, means_B, covs_B, H_init, mask) torch tensors of shapes
        (B, N_max, 2), (B, N_max, 2), (B, N_max, 2, 2), (B, 3, 3), (B, N_max).
        Frames with N < 4 get mask all-zero (the LM cost is constant for
        them; the returned H stays at H_init).
    """
    B = len(per_frame)
    if B == 0:
        raise ValueError("per_frame is empty")
    if len(H_init_per_frame) != B:
        raise ValueError("H_init_per_frame length must equal len(per_frame)")

    Ns = [t[0].shape[0] for t in per_frame]
    N_max = max(Ns) if Ns else 0
    if N_max == 0:
        raise ValueError("All frames have zero correspondences")

    pts_A = torch.zeros((B, N_max, 2), dtype=dtype, device=device)
    means_B = torch.zeros((B, N_max, 2), dtype=dtype, device=device)
    covs_B = torch.zeros((B, N_max, 2, 2), dtype=dtype, device=device)
    H_init = torch.empty((B, 3, 3), dtype=dtype, device=device)
    mask = torch.zeros((B, N_max), dtype=dtype, device=device)

    for i, (pts, mu, _peaks, cov) in enumerate(per_frame):
        n = pts.shape[0]
        if n > 0:
            pts_A[i, :n] = torch.from_numpy(pts.astype(np.float64)).to(device=device, dtype=dtype)
            means_B[i, :n] = torch.from_numpy(mu.astype(np.float64)).to(device=device, dtype=dtype)
            covs_B[i, :n] = torch.from_numpy(cov.astype(np.float64)).to(device=device, dtype=dtype)
            # N<4 cannot constrain a homography. Leave mask zero so the LM
            # treats the cost as constant for this frame and returns H_init
            # unchanged (matches the single-frame API contract).
            if n >= 4:
                mask[i, :n] = 1.0
        H_init[i] = torch.from_numpy(H_init_per_frame[i].astype(np.float64)).to(device=device, dtype=dtype)

    return pts_A, means_B, covs_B, H_init, mask


def pad_for_batched_lm_from_gpu(
    gpu_tensors,
    H_init_per_frame: list,
    dtype: torch.dtype = torch.float64,
):
    """GPU-resident pad: consumes GaussiansBatchTensors from
    find_gaussians_torch_batch(return_tensors=True) without numpy round-trip.

    Only ``H_init_per_frame`` (a list of (3,3) numpy from cv2 RANSAC) crosses
    the host boundary — small (~B*72 bytes), unavoidable because cv2 is CPU.

    Returns (pts_A, means_B, covs_B, H_init, mask) as torch tensors on the
    same device as ``gpu_tensors`` — same shapes as ``pad_for_batched_lm``.
    """
    B = gpu_tensors.B
    counts = gpu_tensors.counts                # (B,) on device
    device = gpu_tensors.pts_A.device

    if len(H_init_per_frame) != B:
        raise ValueError("H_init_per_frame length must equal gpu_tensors.B")

    # We need N_max as a Python int to allocate the padded tensors. Single
    # sync — all the heavy data stays on device.
    N_max = int(counts.max().item()) if counts.numel() > 0 else 0
    if N_max == 0:
        # Every frame had zero correspondences. Caller will short-circuit.
        empty_pts = torch.zeros((B, 0, 2), dtype=dtype, device=device)
        empty_cov = torch.zeros((B, 0, 2, 2), dtype=dtype, device=device)
        empty_mask = torch.zeros((B, 0), dtype=dtype, device=device)
        H_init = torch.from_numpy(
            np.stack(H_init_per_frame).astype(np.float64)
        ).to(device=device, dtype=dtype)
        return empty_pts, empty_pts.clone(), empty_cov, H_init, empty_mask

    pts_A = torch.zeros((B, N_max, 2), dtype=dtype, device=device)
    means_B = torch.zeros((B, N_max, 2), dtype=dtype, device=device)
    covs_B = torch.zeros((B, N_max, 2, 2), dtype=dtype, device=device)
    mask = torch.zeros((B, N_max), dtype=dtype, device=device)

    # batch_idx is sorted (find_gaussians_torch_batch builds it via row-major
    # nonzero), so peaks for frame b live in a contiguous slice. Compute
    # exclusive cumsum once on device.
    starts = torch.cat(
        [
            torch.zeros((1,), dtype=counts.dtype, device=device),
            counts.cumsum(0)[:-1],
        ],
        dim=0,
    )                                          # (B,)

    # Per-frame slot index inside the padded tensor: 0..count[b]-1.
    # Build a flat (N_total,) "slot in frame" vector by subtracting the start
    # offset of each peak's frame.
    slot_in_frame = torch.arange(
        gpu_tensors.batch_idx.shape[0], device=device
    ) - starts[gpu_tensors.batch_idx]          # (N_total,)

    pts_A[gpu_tensors.batch_idx, slot_in_frame] = gpu_tensors.pts_A.to(dtype=dtype)
    means_B[gpu_tensors.batch_idx, slot_in_frame] = gpu_tensors.means_B.to(dtype=dtype)
    covs_B[gpu_tensors.batch_idx, slot_in_frame] = gpu_tensors.covs_B.to(dtype=dtype)
    mask[gpu_tensors.batch_idx, slot_in_frame] = 1.0

    # Frames with N<4 cannot constrain a homography. Zeroing their mask row
    # leaves the LM cost constant for them, so refine_homography_torch_lm_torch
    # returns the H_init we provided (typically identity from the pipeline).
    sparse = counts < 4
    if bool(sparse.any()):
        mask[sparse] = 0.0

    H_init = torch.from_numpy(
        np.stack(H_init_per_frame).astype(np.float64)
    ).to(device=device, dtype=dtype)
    return pts_A, means_B, covs_B, H_init, mask
