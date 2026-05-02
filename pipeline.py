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
from .dlt_ransac import (
    dlt_homography_kornia,
    torch_ransac_homography,
    _KORNIA_OK,
)
from .homography import optimize_homography
from .homography_torch_lm import (
    LMHistory,
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
    """
    H: np.ndarray
    H_init: np.ndarray
    history: Optional["LMHistory"] = None
    per_frame: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None


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
) -> np.ndarray:
    """Run the chosen init backend for a single frame and return a normalized 3x3 H.

    ``init_backend`` selects between cv2.findHomography (default), kornia's
    DLT / IRLS-DLT, or the hand-rolled batched torch RANSAC. The latter
    three are torch-native; we still return a numpy 3x3 here so callers
    that operate in numpy land (legacy ``optimize_homography``) work
    unchanged.
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

    # Torch-native paths. Build a (1, N, 2) batched tensor on CPU; the
    # caller might re-upload to GPU later but for single-frame init this
    # is fine.
    pts_A_t = torch.from_numpy(pts_A.astype(np.float64)).unsqueeze(0)
    pts_B_t = torch.from_numpy(target.astype(np.float64)).unsqueeze(0)

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
            init_backend=init_backend,
        )
        if not refine:
            H = H_init
        else:
            pts_A_t = torch.from_numpy(pts_A.astype(np.float64)).to(device)
            means_B_t = torch.from_numpy(means_B.astype(np.float64)).to(device)
            covs_B_t = torch.from_numpy(covs_B.astype(np.float64)).to(device)
            H_init_t = torch.from_numpy(H_init).to(device)
            with torch.no_grad():
                H_t = refine_homography_torch_lm_torch(
                    pts_A_t, means_B_t, covs_B_t, H_init_t,
                    model=model, f_scale=f_scale, max_iter=max_iter,
                )
            H = H_t[0].detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    if return_details:
        return HomographyResult(H, H_init, pts_A, means_B, peaks_B, covs_B)
    return H


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
    track_history: bool = False,
    logger=None,
    return_per_frame: bool = False,
    return_result: bool = False,
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

    pts_A_cpu = gpu_out.pts_A.detach().cpu().numpy().astype(np.float32, copy=False)
    peaks_B_cpu = gpu_out.peaks_B.detach().cpu().numpy().astype(np.float32, copy=False)
    means_B_cpu = (
        gpu_out.means_B.detach().cpu().numpy().astype(np.float32, copy=False)
        if use_means_for_ransac else None
    )
    starts = np.concatenate([[0], np.cumsum(counts_cpu)[:-1]]).astype(np.int64)

    H_inits: List[np.ndarray] = []
    for b in range(B):
        s = int(starts[b])
        n = int(counts_cpu[b])
        if n < 4:
            H_inits.append(np.eye(3, dtype=np.float64))
        else:
            pts = pts_A_cpu[s:s + n]
            peaks = peaks_B_cpu[s:s + n]
            mu = means_B_cpu[s:s + n] if use_means_for_ransac else None
            H_inits.append(_ransac_init_one(
                pts, peaks, mu, use_means_for_ransac,
                ransac_method, ransac_reproj_threshold,
                ransac_max_iters, ransac_confidence,
                init_backend=init_backend,
            ))

    history: Optional[LMHistory] = None
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
            lm_out = refine_homography_torch_lm_torch(
                pts_A_p, means_B_p, covs_B_p, H_init_p,
                mask=mask, model=model,
                f_scale=f_scale, max_iter=max_iter,
                track_history=track_history, logger=logger,
            )
        if track_history:
            H_t, history = lm_out
        else:
            H_t = lm_out
        H_np = H_t.detach().cpu().numpy()

    # Build per_frame list eagerly when ANY downstream consumer wants it
    # (legacy return_per_frame OR return_result with return_per_frame).
    per_frame_legacy = None
    if return_per_frame:
        per_frame_legacy = []
        covs_B_cpu = gpu_out.covs_B.detach().cpu().numpy().astype(np.float32, copy=False)
        if means_B_cpu is None:
            means_B_cpu = gpu_out.means_B.detach().cpu().numpy().astype(np.float32, copy=False)
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
        )

    extras: List = []
    if return_per_frame:
        extras.append(per_frame_legacy)
    if track_history:
        extras.append(history)
    if not extras:
        return H_np
    return (H_np, *extras)
