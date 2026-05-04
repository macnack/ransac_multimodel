"""Post-refine divergence guard: pure numpy, zero torch dependency.

Detects when LM refinement has degraded (blown up) and falls back to RANSAC init.
Suitable for production deployment on both GPU (cloud) and edge devices.

Key design:
  - Pure numpy. No torch import.
  - Stateless. Guard is a function, not a stateful monitor.
  - Per-sample audit trail. Each sample gets a dict of which checks tripped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DivergenceGuardConfig:
    """Thresholds for detecting and guarding against LM divergence.

    Each threshold is independent; triggering ANY of them causes a fallback
    to H_init. All thresholds are optional (set to None to disable).

    Attributes
    ----------
    max_cost_ratio
        Fallback if final_cost / init_cost > this value (LM made it worse).
        Set to 1.0 (default) to catch any cost increase.
    max_h_diff_fro
        Fallback if ||H_refined - H_init||_F > this value (extreme jump).
        Catches pathological LM updates. Set to None to disable.
    det_min
        Fallback if |det(H_refined)| < this value (degenerate, singular-like).
        Set to None to disable; set to 0.05 (default) for safety.
    det_max
        Fallback if |det(H_refined)| > this value (degenerate, extreme scaling).
        Set to None to disable; set to 20.0 (default) to catch blowup.
    require_converged
        Fallback if not converged AND reached max_iters. Requires history.
        Set to False (default) to skip; True for strict.
    finite_only
        Always fallback on NaN/Inf in H_refined (safety). Default True.
    """

    max_cost_ratio: Optional[float] = 1.0
    max_h_diff_fro: Optional[float] = 5.0
    det_min: Optional[float] = 0.05
    det_max: Optional[float] = 20.0
    require_converged: bool = False
    finite_only: bool = True


@dataclass
class NumpyLMHistory:
    """LM trajectory normalized to numpy (no torch).

    Subset of LMHistory compatible with apply_divergence_guard. Built either
    from torch LMHistory or directly from scipy.optimize.least_squares on edge.

    Attributes
    ----------
    cost_init
        (B,) float64 array. Initial cost per sample (first iteration).
    cost_final
        (B,) float64 array. Final cost per sample.
    n_iters
        int. Actual iterations run (same for all batch elements).
    converged
        (B,) bool array. Convergence flag per sample.
    accept_rate
        (B,) float64 array, optional. Fraction of iterations accepted per sample.
    barrier_active
        (B,) bool array, optional. Whether barrier constraint was active.
    """

    cost_init: np.ndarray  # (B,) float64
    cost_final: np.ndarray  # (B,) float64
    n_iters: int
    converged: np.ndarray  # (B,) bool
    accept_rate: Optional[np.ndarray] = None  # (B,) float64
    barrier_active: Optional[np.ndarray] = None  # (B,) bool


def apply_divergence_guard(
    H_init: np.ndarray,
    H_refined: np.ndarray,
    history: Optional[NumpyLMHistory],
    config: DivergenceGuardConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, bool]]]:
    """Apply post-refine divergence guard.

    Checks each sample in a batch against divergence thresholds. If ANY
    threshold triggers, that sample falls back to H_init. Returns the guarded
    H matrix, a mask of which samples were reverted, and per-sample audit trail.

    Parameters
    ----------
    H_init
        (B, 3, 3) float64 array. Initial homography (RANSAC).
    H_refined
        (B, 3, 3) float64 array. Refined homography (LM output).
    history
        NumpyLMHistory or None. If None, cost/convergence checks are skipped.
    config
        DivergenceGuardConfig. Thresholds and policies.

    Returns
    -------
    H_returned
        (B, 3, 3) float64 array. Guarded result (H_init or H_refined per sample).
    mask_diverged
        (B,) bool array. True where fallback occurred.
    reasons
        list of B dicts. reasons[b] = {"cost_increased": bool, ...}.
        Keys present = checks that were enabled and ran;
        value = whether that check tripped for this sample.
    """
    B = H_init.shape[0]

    mask_diverged = np.zeros(B, dtype=bool)
    reasons: list[dict[str, bool]] = []

    # Always-on safety: finite check.
    finite_mask = np.isfinite(H_refined).all(axis=(1, 2))

    for b in range(B):
        reason = {}

        # 1. Finite check (always on if config.finite_only).
        is_finite = finite_mask[b]
        if config.finite_only:
            reason["finite_only"] = not is_finite
            if not is_finite:
                mask_diverged[b] = True

        # Skip remaining checks if already diverged via finite check.
        if mask_diverged[b]:
            reasons.append(reason)
            continue

        # 2. Cost ratio (requires history).
        if config.max_cost_ratio is not None and history is not None:
            if history.cost_init[b] > 0:
                cost_ratio = history.cost_final[b] / history.cost_init[b]
                trip = cost_ratio > config.max_cost_ratio
            else:
                cost_ratio = 1.0
                trip = False
            reason["cost_increased"] = trip
            if trip:
                mask_diverged[b] = True

        # Skip remaining checks if already diverged via cost.
        if mask_diverged[b]:
            reasons.append(reason)
            continue

        # 3. H frobenius norm difference.
        if config.max_h_diff_fro is not None:
            h_diff = np.linalg.norm(H_refined[b] - H_init[b], ord="fro")
            trip = h_diff > config.max_h_diff_fro
            reason["h_diff_extreme"] = trip
            if trip:
                mask_diverged[b] = True

        # Skip remaining checks if already diverged via H diff.
        if mask_diverged[b]:
            reasons.append(reason)
            continue

        # 4. Determinant bounds.
        det = np.linalg.det(H_refined[b])
        det_out = False
        if config.det_min is not None:
            if np.abs(det) < config.det_min:
                det_out = True
        if config.det_max is not None:
            if np.abs(det) > config.det_max:
                det_out = True
        reason["det_out_of_range"] = det_out
        if det_out:
            mask_diverged[b] = True

        # Skip remaining checks if already diverged via det.
        if mask_diverged[b]:
            reasons.append(reason)
            continue

        # 5. Convergence check (requires history).
        if config.require_converged and history is not None:
            converged = history.converged[b]
            trip = not converged and (history.n_iters > 0)  # hit max_iter without convergence
            reason["not_converged"] = trip
            if trip:
                mask_diverged[b] = True

        reasons.append(reason)

    # Build return H: H_init where mask_diverged, H_refined elsewhere.
    H_returned = np.where(mask_diverged[:, None, None], H_init, H_refined)

    return H_returned, mask_diverged, reasons


# Pre-configured defaults for different deployment scenarios.

#: Strict thresholds for production drone deployment.
#: Safety-critical: fail-closed on any ambiguous sign of trouble.
DEFAULT_GUARD_DRONE = DivergenceGuardConfig(
    max_cost_ratio=1.0,  # LM must improve cost
    max_h_diff_fro=5.0,  # no extreme jumps
    det_min=0.1,  # non-singular
    det_max=10.0,  # no extreme scaling
    require_converged=False,  # don't require convergence (can be slow)
    finite_only=True,  # always check NaN/Inf
)

#: Permissive thresholds for research/tuning.
#: Only catch obvious blow-ups; let marginal cases through for analysis.
DEFAULT_GUARD_RESEARCH = DivergenceGuardConfig(
    max_cost_ratio=2.0,  # allow up to 2x cost increase
    max_h_diff_fro=20.0,  # large jumps OK if cost improved
    det_min=0.01,  # minimal non-singularity check
    det_max=100.0,  # huge scaling OK in research
    require_converged=False,
    finite_only=True,
)
