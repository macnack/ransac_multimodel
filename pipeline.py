"""High-level end-to-end pipeline: logits -> homography.

Two public functions wrap the existing extract -> RANSAC -> refine machinery
so callers don't have to assemble it from
:mod:`ransac_multimodel.correspondence(_torch)`,
:mod:`cv2.findHomography`, and
:mod:`ransac_multimodel.homography(_torch_lm)` themselves.

* :func:`estimate_homography` -- single frame ``(M, in_h, in_w)`` logits.
* :func:`estimate_homography_batched` -- batched ``(B, M, in_h, in_w)``.

Three backends:

* ``"numpy"``     -- :func:`find_gaussians` + scipy
                     :func:`optimize_homography`. CPU only. Reference path.
* ``"torch_cpu"`` -- :func:`find_gaussians_torch` + cv2 RANSAC +
                     :func:`refine_homography_torch_lm_torch` on CPU.
* ``"torch_cuda"`` -- same, on CUDA. Recommended for batched workloads.

The batched path uses :func:`find_gaussians_torch_batch` (one batched conv
pass) and :func:`pad_for_batched_lm` so frames with different N get refined
together with no per-frame Python loop on the LM step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .correspondence import find_gaussians
from .correspondence_torch import find_gaussians_torch, find_gaussians_torch_batch
from .divergence_guard import (
    DivergenceGuardConfig,
    apply_divergence_guard,
)
from .dlt_ransac import (
    dlt_homography_kornia,
    torch_ransac_homography,
    _KORNIA_OK,
)
from .homography import optimize_homography
from .homography_torch_lm import (
    LMHistory,
    lm_history_to_numpy,
    pad_for_batched_lm,
    pad_for_batched_lm_from_gpu,
    refine_homography_torch_lm_torch,
)


# --------------------------------------------------------------------------- #
# CLI-friendly constants and resolvers                                        #
# --------------------------------------------------------------------------- #

# Backend names accepted by estimate_homography{,_batched}. Exported so a
# downstream CLI can do something like:
#
#     parser.add_argument("--backend",
#         choices=list(BACKENDS) + ["auto"], default="auto")
#     args = parser.parse_args()
#     backend = resolve_backend(args.backend)
#     H = estimate_homography(logits, backend=backend)
BACKENDS: Tuple[str, ...] = ("numpy", "torch_cpu", "torch_cuda")

#: Default backend for single-frame inference (CPU, dependency-light).
DEFAULT_BACKEND: str = "numpy"

#: Default backend for batched inference (GPU when available; tune via
#: ``resolve_backend("auto")`` if you want runtime auto-selection).
DEFAULT_BATCHED_BACKEND: str = "torch_cuda"


def resolve_backend(name: str, *, prefer_cuda: bool = True) -> str:
    """Resolve a possibly-symbolic backend name to a concrete one.

    ``"auto"``      -> ``"torch_cuda"`` if ``torch.cuda.is_available()``,
                       else ``"torch_cpu"``.
    ``"torch"``     -> same as ``"auto"``.
    Anything else   -> validated against :data:`BACKENDS` and returned as-is.

    Lets a downstream CLI accept ``--backend auto`` without having to import
    torch and check CUDA itself.
    """
    if name in ("auto", "torch"):
        if prefer_cuda and torch.cuda.is_available():
            return "torch_cuda"
        return "torch_cpu"
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}; choose from {BACKENDS} (or 'auto')."
        )
    return name


# Friendly names for the cv2 RANSAC variants we benchmarked. Keys are what
# you'd put behind a CLI flag; values are the cv2 method ints.
RANSAC_METHODS: dict = {
    "ransac":         cv2.RANSAC,
    "usac_default":   cv2.USAC_DEFAULT,
    "usac_fast":      cv2.USAC_FAST,
    "usac_accurate":  cv2.USAC_ACCURATE,
    "usac_prosac":    cv2.USAC_PROSAC,
    "usac_magsac":    cv2.USAC_MAGSAC,
    "usac_parallel":  cv2.USAC_PARALLEL,
    "lmeds":          cv2.LMEDS,
}

#: Default RANSAC variant (USAC_FAST: 2-4x faster than RANSAC at no
#: accuracy cost on these problem sizes).
DEFAULT_RANSAC_METHOD_NAME: str = "usac_fast"


def resolve_ransac_method(name: str) -> int:
    """Map a CLI-friendly RANSAC variant name to cv2's int constant."""
    key = name.lower()
    if key not in RANSAC_METHODS:
        raise ValueError(
            f"Unknown RANSAC variant {name!r}; choose from {sorted(RANSAC_METHODS)}."
        )
    return RANSAC_METHODS[key]


# Init-backend names accepted by estimate_homography{,_batched}. Picks
# WHICH algorithm produces H_init before the LM refinement.
#
#   "cv2"          - cv2.findHomography(ransac_method=...). CPU only,
#                    not differentiable, sequential per frame, but fast
#                    and outlier-robust. Default.
#   "kornia_dlt"   - kornia weighted DLT. Batched, GPU, differentiable.
#                    No outlier rejection — good when correspondences
#                    are clean.
#   "kornia_dlt_iter" - kornia IRLS-DLT. Batched, GPU, differentiable.
#                       Soft outlier down-weighting via IRLS.
#   "torch_ransac" - hand-rolled batched RANSAC + 4-pt DLT, pure torch.
#                    Batched, GPU, partially differentiable (sample +
#                    inlier-count are non-diff; final DLT solve is).
INIT_BACKENDS: Tuple[str, ...] = ("cv2", "kornia_dlt", "kornia_dlt_iter", "torch_ransac")
DEFAULT_INIT_BACKEND: str = "cv2"


def resolve_init_backend(name: str) -> str:
    """Validate an init-backend name (defensive — useful behind a CLI flag)."""
    if name not in INIT_BACKENDS:
        raise ValueError(
            f"Unknown init backend {name!r}; choose from {INIT_BACKENDS}."
        )
    if name.startswith("kornia") and not _KORNIA_OK:
        raise ModuleNotFoundError(
            f"init_backend={name!r} needs kornia. `pip install kornia` "
            f"or use one of {[b for b in INIT_BACKENDS if not b.startswith('kornia')]}."
        )
    return name


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class HomographyResult:
    """Optional rich return type for callers who want intermediate state.

    Default :func:`estimate_homography` returns just the H matrix; pass
    ``return_details=True`` to get this dataclass instead.

    All fields are populated whenever they're cheap to compute. ``history``
    is only set when the caller asked for it (``track_history=True``); use
    None as a sentinel.
    """
    H: np.ndarray
    H_init: np.ndarray
    pts_A: np.ndarray
    means_B: np.ndarray
    peaks_B: np.ndarray
    covs_B: np.ndarray
    history: Optional["LMHistory"] = None  # set when track_history=True


@dataclass
class BatchedHomographyResult:
    """Forward-compatible return type for :func:`estimate_homography_batched`.

    Designed so callers can opt into fields they care about today without
    rewriting code when they want extra fields later. Example::

        # Today: caller only needs H_init + history.
        res = estimate_homography_batched(
            stacked, backend="torch_cuda",
            return_result=True, refine=True, track_history=True,
        )
        H_init = res.H_init       # (B, 3, 3) — RANSAC init, always present
        hist   = res.history      # LMHistory (only when track_history=True)

        # Tomorrow: same call, also use refined H — no signature change.
        H_final = res.H           # (B, 3, 3) — refined (or H_init if refine=False)

    Fields:
      H        : (B, 3, 3) — refined homography. Equal to H_init when
                 refine=False (so callers can ignore the distinction if
                 they don't need both).
      H_init   : (B, 3, 3) — RANSAC initial guess (always populated).
      history  : Optional[LMHistory] — per-iter LM trajectory; None when
                 track_history=False or refine=False.
      per_frame: Optional[list of (pts_A, means_B, peaks_B, covs_B) numpy
                 tuples] — per-frame extracted correspondences; None when
                 return_per_frame=False.
      mask_diverged: Optional[np.ndarray] — (B,) bool, True where divergence
                 guard reverted to H_init; None when divergence_guard=None.
      guard_reasons: Optional[list[dict]] — per-sample audit trail from
                 divergence guard; None when divergence_guard=None.
    """
    H: np.ndarray
    H_init: np.ndarray
    history: Optional["LMHistory"] = None
    per_frame: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None
    mask_diverged: Optional[np.ndarray] = None  # (B,) bool when guard active
    guard_reasons: Optional[List[dict]] = None  # list of dicts when guard active


def _resolve_torch_device(backend: str) -> str:
    if backend == "torch_cpu":
        return "cpu"
    if backend == "torch_cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("backend='torch_cuda' but torch.cuda.is_available() is False")
        return "cuda"
    raise ValueError(f"Unknown torch backend: {backend!r}")


def _ransac_init_one(
    pts_A: np.ndarray,
    peaks_B: np.ndarray,
    means_B: np.ndarray,
    use_means: bool,
    ransac_method: int,
    reproj: float,
    max_iters: int,
    confidence: float,
    init_backend: str = "cv2",
    device: str = "cpu",
) -> np.ndarray:
    """Run the chosen init backend for a single frame and return a normalized 3x3 H.

    ``init_backend`` selects between cv2.findHomography (default), kornia's
    DLT / IRLS-DLT, or the hand-rolled batched torch RANSAC. The latter
    three are torch-native; we still return a numpy 3x3 here so callers
    that operate in numpy land (legacy ``optimize_homography``) work
    unchanged.

    ``device`` controls where the torch tensors are built for the
    non-cv2 paths. Pass ``"cuda"`` from a CUDA pipeline so kornia /
    torch_ransac runs on GPU instead of paying a host round-trip.
    """
    target = means_B if use_means else (peaks_B if peaks_B is not None else means_B)

    if init_backend == "cv2":
        H, _ = cv2.findHomography(
            pts_A, target, ransac_method,
            ransacReprojThreshold=reproj, maxIters=max_iters, confidence=confidence,
        )
        if H is None:
            return np.eye(3, dtype=np.float64)
        return H / H[2, 2]

    # Torch-native paths. Build (1, N, 2) tensors on the requested device
    # so kornia / torch_ransac actually run on GPU when the caller asked
    # for it (was forcing CPU here, defeating the batched/GPU init point).
    target_device = torch.device(device)
    pts_A_t = torch.from_numpy(pts_A.astype(np.float64)).to(target_device).unsqueeze(0)
    pts_B_t = torch.from_numpy(target.astype(np.float64)).to(target_device).unsqueeze(0)

    with torch.no_grad():
        if init_backend == "kornia_dlt":
            H_t = dlt_homography_kornia(pts_A_t, pts_B_t)[0]
        elif init_backend == "kornia_dlt_iter":
            H_t = dlt_homography_kornia(pts_A_t, pts_B_t, iterated=True)[0]
        elif init_backend == "torch_ransac":
            H_t, _mask = torch_ransac_homography(
                pts_A_t, pts_B_t,
                inlier_threshold=reproj,
                seed=None,  # let caller seed via torch.manual_seed if they want
            )
            H_t = H_t[0]
        else:
            raise ValueError(f"Unknown init_backend: {init_backend!r}")

    H = H_t.cpu().numpy()
    return H / H[2, 2]


def _ransac_init_weighted(
    pts_A: np.ndarray,
    pts_B: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    reproj: float,
    n_iter: int,
    seed: int,
) -> Optional[np.ndarray]:
    """Custom RANSAC with weighted 4-point sampling: P(i) ∝ weights[i]^gamma.

    Used for "p_mode" / "p_mode_power" sampling modes for fair comparison
    against uniform RANSAC. cv2.findHomography(method=0) is DLT — used for
    the per-sample fit because USAC_MAGSAC ignores caller-supplied indices.

    Returns normalized 3x3 H (refit on inliers), or None on degenerate input.
    """
    n = pts_A.shape[0]
    if n < 4:
        return None
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=np.float64) ** float(gamma)
    if w.sum() <= 0 or not np.isfinite(w.sum()):
        p = np.full(n, 1.0 / n)
    else:
        p = w / w.sum()
    pts_A_h = np.hstack([pts_A.astype(np.float32), np.ones((n, 1), dtype=np.float32)])

    best_inliers = -1
    best_H = None
    for _ in range(n_iter):
        idx = rng.choice(n, size=4, replace=False, p=p)
        H, _ = cv2.findHomography(pts_A[idx], pts_B[idx], method=0)
        if H is None:
            continue
        proj = (H @ pts_A_h.T).T
        z = proj[:, 2:3]
        z = np.where(np.abs(z) < 1e-12, 1e-12, z)
        proj2d = proj[:, :2] / z
        d = np.linalg.norm(proj2d - pts_B, axis=1)
        n_inliers = int((d < reproj).sum())
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_H = H
    if best_H is None:
        return None
    # Refit on inliers if 4+.
    proj = (best_H @ pts_A_h.T).T
    z = proj[:, 2:3]
    z = np.where(np.abs(z) < 1e-12, 1e-12, z)
    d = np.linalg.norm(proj[:, :2] / z - pts_B, axis=1)
    inlier_mask = d < reproj
    if int(inlier_mask.sum()) >= 4:
        H_refit, _ = cv2.findHomography(pts_A[inlier_mask], pts_B[inlier_mask], method=0)
        if H_refit is not None:
            best_H = H_refit
    return best_H / best_H[2, 2]


# --------------------------------------------------------------------------- #
# Public API: single frame                                                    #
# --------------------------------------------------------------------------- #


def estimate_homography(
    logits: torch.Tensor,
    *,
    backend: str = "numpy",
    model: str = "sRT",
    use_means_for_ransac: bool = False,
    init_backend: str = DEFAULT_INIT_BACKEND,
    ransac_method: int = cv2.USAC_FAST,
    ransac_reproj_threshold: float = 3.0,
    ransac_max_iters: int = 5000,
    ransac_confidence: float = 0.995,
    fixed_threshold: float = 0.008,
    fixed_window_size: int = 4,
    f_scale: float = 2.0,
    max_iter: int = 100,
    refine: bool = True,
    lm_kwargs: Optional[dict] = None,
    return_details: bool = False,
) -> Union[np.ndarray, HomographyResult]:
    """End-to-end refinement for a single ``(M, in_h, in_w)`` logit tensor.

    Returns a ``(3, 3)`` numpy float64 homography (feature-grid space) by
    default; pass ``return_details=True`` to get a :class:`HomographyResult`
    that also carries the RANSAC init H and the extracted correspondences.

    Parameters
    ----------
    logits
        Raw per-patch logits; first dim must be a perfect square.
    backend
        ``"numpy"`` (default), ``"torch_cpu"``, or ``"torch_cuda"``.
    model
        ``"sRT"`` or ``"full"`` (passed to the refinement step).
    ransac_method
        ``cv2.RANSAC``, ``cv2.USAC_FAST`` (default), ``cv2.USAC_MAGSAC``,
        etc. ``USAC_FAST`` is 2-4x faster than ``RANSAC`` at no accuracy
        cost on this problem size.
    refine
        If ``True`` (default), run the LM refinement after RANSAC. If
        ``False``, skip the LM step and return the RANSAC init H. Useful
        when you want a cheap estimate (RANSAC alone is ~0.1ms vs ~25ms
        for LM at single-frame, batched at higher B's).
    lm_kwargs
        Optional dict of extra keyword arguments forwarded verbatim to
        :func:`refine_homography_torch_lm_torch` (init_damping, damping_up,
        damping_down, barrier_k, srt_bounds_low, srt_bounds_high,
        abs_err_tolerance, rel_err_tolerance, ...). Only takes effect on
        torch backends with ``refine=True``; ignored on the numpy/scipy
        path. Putting a key already in the explicit signature
        (``f_scale``, ``max_iter``, ``model``) raises
        ``TypeError: multiple values for keyword argument``.
    """
    if backend == "numpy":
        pts_A, means_B, peaks_B, covs_B = find_gaussians(
            logits,
            adaptive_gauss_fit=False,
            log_missing_gaussians=False,
            fixed_threshold=fixed_threshold,
            fixed_window_size=fixed_window_size,
        )
        if pts_A.shape[0] < 4:
            H_eye = np.eye(3, dtype=np.float64)
            if return_details:
                return HomographyResult(H_eye, H_eye, pts_A, means_B, peaks_B, covs_B)
            return H_eye

        if refine:
            if init_backend != "cv2":
                # scipy's optimize_homography does its own cv2 RANSAC init
                # internally and doesn't accept an externally-computed H_init.
                # Honoring init_backend here would require either threading
                # an H_init through scipy's TRF wrapper (invasive change to
                # the legacy code) or doing init outside and refining via
                # the torch-LM (defeats the point of backend="numpy"). Fail
                # loudly rather than silently ignore the kwarg.
                raise ValueError(
                    f"backend='numpy' with refine=True hardwires init via "
                    f"cv2.findHomography (scipy.optimize.least_squares does "
                    f"the chaining internally). init_backend={init_backend!r} "
                    f"is not honored on this path. Either set "
                    f"refine=False (returns the chosen init), or switch "
                    f"backend to 'torch_cpu' / 'torch_cuda' (these honor "
                    f"init_backend end-to-end)."
                )
            H, H_init = optimize_homography(
                pts_A, means_B, covs_B, peaks_B=peaks_B,
                model=model, verbose=0, quiet=True,
                use_means_for_ransac=use_means_for_ransac,
                ransac_method=ransac_method,
                ransac_reproj_threshold=ransac_reproj_threshold,
                ransac_max_iters=ransac_max_iters,
                ransac_confidence=ransac_confidence,
                f_scale=f_scale,
            )
        else:
            H_init = _ransac_init_one(
                pts_A, peaks_B, means_B, use_means_for_ransac,
                ransac_method, ransac_reproj_threshold,
                ransac_max_iters, ransac_confidence,
                init_backend=init_backend,
            )
            if H_init is None:
                H_init = np.eye(3, dtype=np.float64)
            H = H_init

    elif backend in ("torch_cpu", "torch_cuda"):
        device = _resolve_torch_device(backend)
        logits_dev = logits.to(device, non_blocking=True)
        pts_A, means_B, peaks_B, covs_B = find_gaussians_torch(
            logits_dev,
            adaptive_gauss_fit=False,
            log_missing_gaussians=False,
            fixed_threshold=fixed_threshold,
            fixed_window_size=fixed_window_size,
            device=device,
        )
        if pts_A.shape[0] < 4:
            H_eye = np.eye(3, dtype=np.float64)
            if return_details:
                return HomographyResult(H_eye, H_eye, pts_A, means_B, peaks_B, covs_B)
            return H_eye

        H_init = _ransac_init_one(
            pts_A, peaks_B, means_B, use_means_for_ransac,
            ransac_method, ransac_reproj_threshold,
            ransac_max_iters, ransac_confidence,
            init_backend=init_backend, device=device,
        )
        if H_init is None:
            H_init = np.eye(3, dtype=np.float64)
        if not refine:
            H = H_init
        else:
            pts_A_t = torch.from_numpy(pts_A.astype(np.float64)).to(device)
            means_B_t = torch.from_numpy(means_B.astype(np.float64)).to(device)
            covs_B_t = torch.from_numpy(covs_B.astype(np.float64)).to(device)
            H_init_t = torch.from_numpy(H_init).to(device)
            with torch.no_grad():
                _kw = dict(lm_kwargs or {})
                _f_scale = _kw.pop("f_scale", f_scale)
                _max_iter = _kw.pop("max_iter", max_iter)
                H_t = refine_homography_torch_lm_torch(
                    pts_A_t, means_B_t, covs_B_t, H_init_t,
                    model=model, f_scale=_f_scale, max_iter=_max_iter,
                    **_kw,
                )
            H = H_t[0].detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    if return_details:
        return HomographyResult(H, H_init, pts_A, means_B, peaks_B, covs_B)
    return H


def _batched_init_from_gpu(
    *,
    gpu_out,
    counts_cpu: np.ndarray,
    starts: np.ndarray,
    init_backend: str,
    use_means: bool,
    reproj: float,
    device: str,
) -> List[np.ndarray]:
    """Run a non-cv2 init backend on padded GPU tensors in ONE batched call.

    Bypasses the per-frame Python loop that the cv2 path is forced into
    (cv2 has no batched API). Pads gpu_out's per-frame correspondence
    slices to (B, N_max, 2), runs kornia DLT / kornia IRLS / batched
    torch RANSAC once, returns a list of B numpy 3x3 H matrices.

    Frames with N<4 get identity H (RANSAC over <4 pts is undefined).
    """
    B = gpu_out.B
    N_max = int(counts_cpu.max()) if counts_cpu.size > 0 else 0
    if N_max < 4:
        return [np.eye(3, dtype=np.float64) for _ in range(B)]

    # Pad pts_A and the RANSAC target on device using the gpu_out batch_idx.
    target_dev = torch.device(device)
    dtype = torch.float64

    pts_src = gpu_out.pts_A.to(dtype=dtype)               # (N_total, 2)
    pts_tgt_full = (
        gpu_out.means_B.to(dtype=dtype) if use_means
        else gpu_out.peaks_B.to(dtype=dtype)
    )                                                     # (N_total, 2)

    counts_dev = gpu_out.counts.to(target_dev)            # (B,)
    starts_dev = torch.cat(
        [torch.zeros(1, dtype=counts_dev.dtype, device=target_dev),
         counts_dev.cumsum(0)[:-1]],
    )                                                     # (B,)
    slot_in_frame = torch.arange(
        gpu_out.batch_idx.shape[0], device=target_dev,
    ) - starts_dev[gpu_out.batch_idx]                     # (N_total,)

    pts_A_p = torch.zeros((B, N_max, 2), dtype=dtype, device=target_dev)
    pts_B_p = torch.zeros((B, N_max, 2), dtype=dtype, device=target_dev)
    weights = torch.zeros((B, N_max), dtype=dtype, device=target_dev)
    pts_A_p[gpu_out.batch_idx, slot_in_frame] = pts_src
    pts_B_p[gpu_out.batch_idx, slot_in_frame] = pts_tgt_full
    weights[gpu_out.batch_idx, slot_in_frame] = 1.0

    # Frames with N<4: zero-out their weights so kornia DLT ignores them
    # (we'll overwrite their H with identity below).
    sparse = (counts_dev < 4)
    if bool(sparse.any()):
        weights[sparse] = 0.0

    with torch.no_grad():
        if init_backend == "kornia_dlt":
            # Weighted DLT — weights act as RANSAC inlier mask.
            H_t = dlt_homography_kornia(pts_A_p, pts_B_p, weights=weights)
        elif init_backend == "kornia_dlt_iter":
            H_t = dlt_homography_kornia(
                pts_A_p, pts_B_p, weights=weights, iterated=True,
            )
        elif init_backend == "torch_ransac":
            H_t, _mask = torch_ransac_homography(
                pts_A_p, pts_B_p,
                inlier_threshold=reproj,
                seed=None,
            )
        else:
            raise ValueError(f"Unknown init_backend: {init_backend!r}")

    H_np = H_t.detach().cpu().numpy().astype(np.float64)
    # Restore identity for sparse frames (kornia/torch_ransac result on
    # all-zero-weight rows is meaningless).
    sparse_np = (counts_cpu < 4)
    H_inits: List[np.ndarray] = []
    eye = np.eye(3, dtype=np.float64)
    for b in range(B):
        if sparse_np[b]:
            H_inits.append(eye.copy())
        else:
            h = H_np[b]
            H_inits.append(h / h[2, 2])
    return H_inits


# --------------------------------------------------------------------------- #
# Public API: batched                                                         #
# --------------------------------------------------------------------------- #


def estimate_homography_batched(
    logits: torch.Tensor,
    *,
    backend: str = "torch_cuda",
    model: str = "sRT",
    use_means_for_ransac: bool = False,
    init_backend: str = DEFAULT_INIT_BACKEND,
    ransac_method: int = cv2.USAC_FAST,
    ransac_reproj_threshold: float = 3.0,
    ransac_max_iters: int = 5000,
    ransac_confidence: float = 0.995,
    fixed_threshold: float = 0.008,
    fixed_window_size: int = 4,
    f_scale: float = 2.0,
    max_iter: int = 100,
    refine: bool = True,
    lm_kwargs: Optional[dict] = None,
    track_history: bool = False,
    logger=None,
    return_per_frame: bool = False,
    return_result: bool = False,
    divergence_guard: Optional[DivergenceGuardConfig] = None,
    ransac_sampling: str = "uniform",
    ransac_sampling_gamma: float = 1.0,
    ransac_sampling_iters: int = 2000,
    ransac_sampling_seed: int = 0,
):
    """Batched end-to-end refinement for ``(B, M, in_h, in_w)`` logits.

    Returns a ``(B, 3, 3)`` numpy float64 array of feature-grid-space H
    matrices. Pass ``return_per_frame=True`` to also get the list of
    extracted ``(pts_A, means_B, peaks_B, covs_B)`` numpy tuples.

    Internally:

    1. ``find_gaussians_torch_batch`` does extraction in one batched conv pass.
    2. cv2 RANSAC runs in a Python loop (cv2 has no batched API; cheap with
       ``USAC_FAST`` -- ~3 ms total at B=64).
    3. ``pad_for_batched_lm_from_gpu`` packs the variable-N per-frame outputs
       into padded tensors with a mask, and ``refine_homography_torch_lm_torch``
       runs the LM step batched. Frames with N<4 get their RANSAC init back
       (the LM cost is constant for them under the all-zero mask).

    The ``"numpy"`` backend exists for parity testing -- it loops over B
    sequentially and gives no batching benefit.

    Parameters
    ----------
    refine
        If ``True`` (default), run the batched LM refinement after RANSAC.
        If ``False``, skip the LM step and return the RANSAC inits stacked.
        Skipping LM at B=64 saves ~30-50 ms (the LM is the dominant cost
        once extraction is batched).
    return_per_frame
        If ``True``, also return a list of B per-frame
        ``(pts_A, means_B, peaks_B, covs_B)`` numpy tuples (one per input
        frame, even when a frame had zero detections).
    return_result
        If ``True``, return a single :class:`BatchedHomographyResult`
        dataclass with H, H_init, history, per_frame fields. Future-proof
        alternative to the tuple-return modes (which grow in arity as more
        opt-in fields land — calling code stays unchanged when you start
        accessing more fields on the dataclass).
    lm_kwargs
        Optional dict of extra keyword arguments forwarded verbatim to
        :func:`refine_homography_torch_lm_torch` on the torch backends
        (e.g. ``{"init_damping": 1e-2, "damping_up": 3.0, "barrier_k": 2.0,
        "abs_err_tolerance": 1e-9}``). Same caveats as on the single-frame
        version: torch + ``refine=True`` only; conflicts with explicit
        params raise ``TypeError``.
    divergence_guard
        Optional :class:`DivergenceGuardConfig`. If set, post-refine H is
        checked for divergence (cost increase, extreme jumps, degenerate det).
        Samples that diverge fall back to H_init. Only applies when
        ``refine=True``. When enabled, ``track_history=True`` is promoted
        internally (guard needs cost trajectory). Results include ``mask_diverged``
        and ``guard_reasons`` fields in BatchedHomographyResult.
    """
    if logits.dim() != 4:
        raise ValueError(f"Batched logits must be (B, M, in_h, in_w), got {tuple(logits.shape)}")
    B = logits.shape[0]

    # Empty per-frame placeholders shared by every short-circuit / fallback
    # path so the return_per_frame contract always yields exactly B tuples.
    _empty_pts = np.zeros((0, 2), dtype=np.float32)
    _empty_cov = np.zeros((0, 2, 2), dtype=np.float32)

    def _empty_per_frame_list() -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        return [
            (_empty_pts.copy(), _empty_pts.copy(), _empty_pts.copy(), _empty_cov.copy())
            for _ in range(B)
        ]

    if backend == "numpy":
        # Sequential reference path (slow but exists for parity tests).
        results = np.zeros((B, 3, 3), dtype=np.float64)
        results_init = np.zeros((B, 3, 3), dtype=np.float64)
        per_frame_out = []
        for b in range(B):
            res = estimate_homography(
                logits[b], backend="numpy", model=model,
                use_means_for_ransac=use_means_for_ransac,
                init_backend=init_backend,
                ransac_method=ransac_method,
                ransac_reproj_threshold=ransac_reproj_threshold,
                ransac_max_iters=ransac_max_iters,
                ransac_confidence=ransac_confidence,
                fixed_threshold=fixed_threshold,
                fixed_window_size=fixed_window_size,
                f_scale=f_scale, max_iter=max_iter,
                refine=refine,
                lm_kwargs=lm_kwargs,
                return_details=True,
            )
            results[b] = res.H
            results_init[b] = res.H_init
            per_frame_out.append((res.pts_A, res.means_B, res.peaks_B, res.covs_B))

        if return_result:
            return BatchedHomographyResult(
                H=results, H_init=results_init,
                history=None,  # numpy backend doesn't surface LM history
                per_frame=per_frame_out if return_per_frame else None,
                mask_diverged=None,  # divergence_guard not supported on numpy backend
                guard_reasons=None,
            )
        if return_per_frame:
            return results, per_frame_out
        return results

    if backend not in ("torch_cpu", "torch_cuda"):
        raise ValueError(f"Unknown backend: {backend!r}")

    device = _resolve_torch_device(backend)
    logits_dev = logits.to(device, non_blocking=True)

    # 1. Batched extraction. Stay on device — only the small (pts_A, peaks_B)
    # subset crosses the host boundary for cv2 RANSAC. means_B + covs_B never
    # leave the GPU, eliminating the GPU→CPU→GPU round-trip the original
    # numpy-list path forced.
    gpu_out = find_gaussians_torch_batch(
        logits_dev,
        fixed_threshold=fixed_threshold,
        fixed_window_size=fixed_window_size,
        device=device,
        return_tensors=True,
    )

    # 2. Per-frame RANSAC. cv2 is CPU-only and not batched; transfer ONLY
    # the inputs RANSAC needs (pts_A + peaks_B, two (N_total, 2) float32
    # arrays — typically ~3 KB at B=64). Per-frame slicing in numpy uses
    # the device-resident counts to avoid scanning batch_idx on host.
    counts_cpu = gpu_out.counts.cpu().numpy()
    if counts_cpu.size == 0 or counts_cpu.max() == 0:
        results = np.tile(np.eye(3, dtype=np.float64), (B, 1, 1))
        results_init = results.copy()
        empty_pf = _empty_per_frame_list()
        if return_result:
            return BatchedHomographyResult(
                H=results, H_init=results_init,
                history=None,
                per_frame=empty_pf if return_per_frame else None,
            )
        if return_per_frame:
            return results, empty_pf
        return results

    starts = np.concatenate([[0], np.cumsum(counts_cpu)[:-1]]).astype(np.int64)

    if ransac_sampling not in ("uniform", "p_mode", "p_mode_power"):
        raise ValueError(
            f"ransac_sampling must be one of 'uniform', 'p_mode', 'p_mode_power'; "
            f"got {ransac_sampling!r}."
        )

    failed_frames: List[int] = []  # populated by cv2 path; empty for GPU init paths

    # Cached host copies — populated lazily by the cv2 init path or by the
    # return_per_frame block below. None until first transfer.
    pts_A_cpu: Optional[np.ndarray] = None
    peaks_B_cpu: Optional[np.ndarray] = None
    means_B_cpu: Optional[np.ndarray] = None

    if init_backend == "cv2":
        # cv2 path: must transfer pts_A + peaks_B (and means_B if used) to
        # CPU since cv2 has no GPU/batched API; loop per frame.
        pts_A_cpu = gpu_out.pts_A.detach().cpu().numpy().astype(np.float32, copy=False)
        peaks_B_cpu = gpu_out.peaks_B.detach().cpu().numpy().astype(np.float32, copy=False)
        means_B_cpu = (
            gpu_out.means_B.detach().cpu().numpy().astype(np.float32, copy=False)
            if use_means_for_ransac else None
        )
        if ransac_sampling != "uniform":
            pv = getattr(gpu_out, "peak_values", None)
            peak_values_cpu = (
                pv.detach().cpu().numpy().astype(np.float32, copy=False)
                if pv is not None
                else np.ones(pts_A_cpu.shape[0], dtype=np.float32)
            )
        else:
            peak_values_cpu = None

        H_inits: List[np.ndarray] = []
        for b in range(B):
            s = int(starts[b])
            n = int(counts_cpu[b])
            if n < 4:
                H_inits.append(np.eye(3, dtype=np.float64))
                failed_frames.append(b)
            else:
                pts = pts_A_cpu[s:s + n]
                peaks = peaks_B_cpu[s:s + n]
                mu = means_B_cpu[s:s + n] if use_means_for_ransac else None
                if ransac_sampling == "uniform":
                    H_init_b = _ransac_init_one(
                        pts, peaks, mu, use_means_for_ransac,
                        ransac_method, ransac_reproj_threshold,
                        ransac_max_iters, ransac_confidence,
                        init_backend="cv2",
                    )
                else:
                    target = mu if use_means_for_ransac else peaks
                    H_init_b = _ransac_init_weighted(
                        pts, target, peak_values_cpu[s:s + n],
                        gamma=ransac_sampling_gamma,
                        reproj=ransac_reproj_threshold,
                        n_iter=ransac_sampling_iters,
                        seed=ransac_sampling_seed + b,
                    )
                if H_init_b is None:
                    H_inits.append(np.eye(3, dtype=np.float64))
                    failed_frames.append(b)
                else:
                    H_inits.append(H_init_b)
    else:
        # GPU-resident batched init: stay on device, use the chosen torch
        # backend in ONE batched call. Skips the GPU→CPU per-frame round
        # trip cv2 forces.
        H_inits = _batched_init_from_gpu(
            gpu_out=gpu_out,
            counts_cpu=counts_cpu,
            starts=starts,
            init_backend=init_backend,
            use_means=use_means_for_ransac,
            reproj=ransac_reproj_threshold,
            device=device,
        )

    history: Optional[LMHistory] = None
    mask_diverged = None
    guard_reasons = None

    # If divergence guard is active, we need history to check cost trajectory
    need_history_for_guard = divergence_guard is not None and refine
    effective_track_history = track_history or need_history_for_guard

    if not refine:
        # Skip LM, just return the stacked RANSAC inits.
        H_np = np.stack(H_inits).astype(np.float64)
    else:
        # 3. Pad on device — means_B + covs_B never left it. Only H_init crosses
        # the host boundary (B * 9 * 8 bytes — negligible).
        pts_A_p, means_B_p, covs_B_p, H_init_p, mask = pad_for_batched_lm_from_gpu(
            gpu_out, H_inits, dtype=torch.float64,
        )
        with torch.no_grad():
            _kw = dict(lm_kwargs or {})
            _f_scale = _kw.pop("f_scale", f_scale)
            _max_iter = _kw.pop("max_iter", max_iter)
            lm_out = refine_homography_torch_lm_torch(
                pts_A_p, means_B_p, covs_B_p, H_init_p,
                mask=mask, model=model,
                f_scale=_f_scale, max_iter=_max_iter,
                track_history=effective_track_history, logger=logger,
                **_kw,
            )
        if effective_track_history:
            H_t, history = lm_out
        else:
            H_t = lm_out
        H_np = H_t.detach().cpu().numpy()

        # Overwrite failed frames with NaN so downstream aggregation treats them
        # as failures rather than successes (eye(3) was only a placeholder for LM).
        if failed_frames:
            nan_mat = np.full((3, 3), np.nan, dtype=H_np.dtype)
            for b in failed_frames:
                H_np[b] = nan_mat

        # Apply divergence guard if configured
        if divergence_guard is not None and history is not None:
            H_init_np = np.stack(H_inits).astype(np.float64)
            history_np = lm_history_to_numpy(history)
            H_np, mask_diverged, guard_reasons = apply_divergence_guard(
                H_init_np, H_np, history_np, divergence_guard
            )

    # Build per_frame list eagerly when ANY downstream consumer wants it
    # (legacy return_per_frame OR return_result with return_per_frame).
    per_frame_legacy = None
    if return_per_frame:
        per_frame_legacy = []
        # Lazy bulk transfers — the GPU-resident init path may have skipped
        # them so far; fill in only what's missing.
        if pts_A_cpu is None:
            pts_A_cpu = gpu_out.pts_A.detach().cpu().numpy().astype(np.float32, copy=False)
        if peaks_B_cpu is None:
            peaks_B_cpu = gpu_out.peaks_B.detach().cpu().numpy().astype(np.float32, copy=False)
        if means_B_cpu is None:
            means_B_cpu = gpu_out.means_B.detach().cpu().numpy().astype(np.float32, copy=False)
        covs_B_cpu = gpu_out.covs_B.detach().cpu().numpy().astype(np.float32, copy=False)
        for b in range(B):
            s = int(starts[b])
            n = int(counts_cpu[b])
            per_frame_legacy.append((
                pts_A_cpu[s:s + n], means_B_cpu[s:s + n],
                peaks_B_cpu[s:s + n], covs_B_cpu[s:s + n],
            ))

    if return_result:
        # Forward-compatible single-object return.
        H_init_np = np.stack(H_inits).astype(np.float64)
        return BatchedHomographyResult(
            H=H_np, H_init=H_init_np,
            history=history, per_frame=per_frame_legacy,
            mask_diverged=mask_diverged, guard_reasons=guard_reasons,
        )

    extras: List = []
    if return_per_frame:
        extras.append(per_frame_legacy)
    if track_history:
        extras.append(history)
    if not extras:
        return H_np
    return (H_np, *extras)
