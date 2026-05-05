"""Vectorized Torch implementation of ``find_gaussians``.

This module mirrors the public behaviour of
:func:`ransac_multimodel.correspondence.find_gaussians` but processes every
patch of the input tensor in a single batched torch pipeline (softmax →
Gaussian blur → local-maxima detection → image-moments based mean / covariance).

The reference numpy/cv2 implementation in ``correspondence.py`` (and its
helpers in ``gaussian_fit.py``) iterates one (py, px) patch at a time and uses
``cv2.GaussianBlur``, ``cv2.dilate`` and ``cv2.moments``.  Here we replicate
each of those operations with their batched ``torch.nn.functional`` analogues.

Returned arrays have the exact same shape semantics as the numpy version:

* ``pts_A``    -- ``(N, 2)`` float32, image-A patch coordinates ``(px, py)``.
* ``means_B``  -- ``(N, 2)`` float32, fitted Gaussian centre in image-B feature
  grid coordinates ``(x, y)``.
* ``peaks_B``  -- ``(N, 2)`` float32, the *global* peak of the patch (matches
  the reference ``g["global_peak"]`` value, which is identical for every
  Gaussian belonging to the same patch).
* ``covs_B``   -- ``(N, 2, 2)`` float32, the per-Gaussian covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _cv2_gaussian_kernel_5x5(sigma: float, device, dtype) -> torch.Tensor:
    """Build the 5x5 separable Gaussian kernel that matches ``cv2.GaussianBlur``.

    For ``ksize=5`` and ``sigma=0`` cv2 derives the sigma from the kernel size:
    ``sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8 = 1.1``.

    cv2 also normalises the 1D kernel so that it sums to 1 (rather than using the
    truncated-Gaussian density), so we mirror that here.
    """
    k = 5
    half = k // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d


def _softmax_reshape_heatmaps(
    logits: torch.Tensor,
    out_patch_size: int,
    in_h: int,
    in_w: int,
) -> torch.Tensor:
    """Apply softmax and reshape logits to heatmaps (single-frame path).
    
    Args:
        logits: (M, in_h, in_w) tensor
        out_patch_size: size of the output patch grid (O = sqrt(M))
        in_h, in_w: spatial dimensions
    
    Returns:
        heatmaps: (n_patches, O, O) tensor
    """
    soft = torch.softmax(logits, dim=0)  # (M, in_h, in_w)
    soft = soft.permute(1, 2, 0).contiguous()  # (in_h, in_w, M)
    n_patches = in_h * in_w
    heatmaps = soft.view(n_patches, out_patch_size, out_patch_size)
    return heatmaps


def _compute_global_peaks(
    heatmaps: torch.Tensor,
    out_patch_size: int,
    work_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute global peak per patch (pre-blur).
    
    Args:
        heatmaps: (n_patches, O, O)
        out_patch_size: O
        work_dtype: torch dtype
    
    Returns:
        global_peaks_xy: (n_patches, 2) with [x, y] coords
    """
    flat = heatmaps.reshape(heatmaps.shape[0], -1)
    global_peak_idx = torch.argmax(flat, dim=1)
    global_peak_y = (global_peak_idx // out_patch_size).to(work_dtype)
    global_peak_x = (global_peak_idx % out_patch_size).to(work_dtype)
    global_peaks_xy = torch.stack([global_peak_x, global_peak_y], dim=1)
    return global_peaks_xy


def _apply_blur_and_detect_peaks(
    heatmaps: torch.Tensor,
    fixed_threshold: float,
    fixed_window_size: int,
    target_device,
    work_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Gaussian blur and max-pool to detect local maxima.
    
    Args:
        heatmaps: (n_patches, O, O)
        fixed_threshold: threshold for peak detection
        fixed_window_size: window size for dilation
        target_device: torch device
        work_dtype: torch dtype
    
    Returns:
        (blurred, peaks_mask) tensors
    """
    blur_kernel = _cv2_gaussian_kernel_5x5(
        sigma=1.1, device=target_device, dtype=work_dtype
    )
    blur_kernel = blur_kernel.view(1, 1, 5, 5)
    heatmaps_padded = F.pad(heatmaps.unsqueeze(1), (2, 2, 2, 2), mode="replicate")
    blurred = F.conv2d(heatmaps_padded, blur_kernel, padding=0)
    
    W = int(fixed_window_size)
    pad_max = W // 2
    pooled = F.max_pool2d(blurred, kernel_size=W, stride=1, padding=pad_max)
    
    if pooled.shape[-1] != blurred.shape[-1]:
        pooled = pooled[..., :blurred.shape[-2], :blurred.shape[-1]]
    
    peaks_mask = (blurred > fixed_threshold) & (blurred == pooled)
    peaks_mask = peaks_mask.squeeze(1)
    
    return blurred, peaks_mask


def _gather_and_compute_moments(
    blurred: torch.Tensor,
    peaks_mask: torch.Tensor,
    fixed_window_size: int,
    out_patch_size: int,
    target_device,
    work_dtype,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Gather windows at peak locations and compute image moments.
    
    Returns:
        (patch_idx, py_peak, px_peak,
         local_mean_x, local_mean_y,
         cov_xx, cov_yy, cov_xy,
         x_min_clamped_global, y_min_clamped_global)
    """
    W = int(fixed_window_size)
    half_w = W // 2
    full_window = 2 * half_w + 1
    
    blurred_padded = F.pad(blurred, (half_w, half_w, half_w, half_w))
    
    coords = torch.arange(out_patch_size, device=target_device, dtype=work_dtype)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    x_min_clamped = (grid_x - half_w).clamp(min=0)
    y_min_clamped = (grid_y - half_w).clamp(min=0)
    
    # Locate peaks
    patch_idx, py_peak, px_peak = peaks_mask.nonzero(as_tuple=True)
    
    # Gather windows
    dy = torch.arange(full_window, device=target_device)
    dx = torch.arange(full_window, device=target_device)
    dy_grid, dx_grid = torch.meshgrid(dy, dx, indexing="ij")
    
    py_idx = py_peak[:, None, None] + dy_grid[None]
    px_idx = px_peak[:, None, None] + dx_grid[None]
    windows = blurred_padded[patch_idx[:, None, None], 0, py_idx, px_idx]
    
    # Compute moments
    coord = torch.arange(full_window, device=target_device, dtype=work_dtype)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    
    m00 = windows.sum(dim=(-1, -2))
    m10 = (windows * xx).sum(dim=(-1, -2))
    m01 = (windows * yy).sum(dim=(-1, -2))
    m11 = (windows * xx * yy).sum(dim=(-1, -2))
    m20 = (windows * xx * xx).sum(dim=(-1, -2))
    m02 = (windows * yy * yy).sum(dim=(-1, -2))
    
    # Filter out degenerate windows
    keep = m00 > 0
    if not bool(keep.all()):
        patch_idx = patch_idx[keep]
        py_peak = py_peak[keep]
        px_peak = px_peak[keep]
        m00 = m00[keep]; m10 = m10[keep]; m01 = m01[keep]
        m11 = m11[keep]; m20 = m20[keep]; m02 = m02[keep]
    
    # Compute mean and covariance
    local_mean_x = m10 / m00
    local_mean_y = m01 / m00
    cov_xx = m20 / m00 - local_mean_x * local_mean_x
    cov_yy = m02 / m00 - local_mean_y * local_mean_y
    cov_xy = m11 / m00 - local_mean_x * local_mean_y
    
    return (
        patch_idx, py_peak, px_peak,
        local_mean_x, local_mean_y,
        cov_xx, cov_yy, cov_xy,
        x_min_clamped, y_min_clamped,
    )


def _assemble_torch_correspondences(
    patch_idx: torch.Tensor,
    py_peak: torch.Tensor,
    px_peak: torch.Tensor,
    local_mean_x: torch.Tensor,
    local_mean_y: torch.Tensor,
    cov_xx: torch.Tensor,
    cov_yy: torch.Tensor,
    cov_xy: torch.Tensor,
    x_min_clamped: torch.Tensor,
    y_min_clamped: torch.Tensor,
    global_peaks_xy: torch.Tensor,
    in_h: int,
    in_w: int,
    work_dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble final correspondence arrays from computed values.
    
    Returns:
        (pts_A, means_B, peaks_B, covs_B) as torch tensors
    """
    # Add window offset to local moments
    sel_mean_x = x_min_clamped[py_peak, px_peak] + local_mean_x
    sel_mean_y = y_min_clamped[py_peak, px_peak] + local_mean_y
    sel_cov_xx = cov_xx + 1e-4
    sel_cov_yy = cov_yy + 1e-4
    sel_cov_xy = cov_xy
    
    # Map patch index to image-A (px, py) coordinates
    patch_py = (patch_idx // in_w).to(work_dtype)
    patch_px = (patch_idx % in_w).to(work_dtype)
    pts_A = torch.stack([patch_px, patch_py], dim=1)
    
    means_B = torch.stack([sel_mean_x, sel_mean_y], dim=1)
    peaks_B = global_peaks_xy[patch_idx]
    
    covs_B = torch.stack(
        [
            torch.stack([sel_cov_xx, sel_cov_xy], dim=1),
            torch.stack([sel_cov_xy, sel_cov_yy], dim=1),
        ],
        dim=1,
    )
    
    return pts_A, means_B, peaks_B, covs_B

    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d


def _moment_kernels(window_size: int, device, dtype) -> torch.Tensor:
    """Build the six raw-moment convolution kernels.

    Returns a tensor of shape ``(6, 1, window_size, window_size)`` whose
    channels correspond to ``[1, x, y, x*y, x*x, y*y]`` evaluated at every pixel
    of the window with the origin at the window's top-left corner (i.e. the
    same convention as ``cv2.moments`` for an image whose top-left is ``(0,0)``).
    """
    half_w = window_size // 2
    # cv2.moments treats the window as an image whose own pixel coordinates run
    # from (0, 0) at the top-left to (window_size - 1, window_size - 1) at the
    # bottom-right.  We compute the moments inside the window in those local
    # coordinates and add the window's top-left offset back when extracting the
    # mean / covariance.
    coords = torch.arange(window_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # (W, W)

    ones = torch.ones_like(xx)
    k_m00 = ones
    k_m10 = xx                  # sum of x
    k_m01 = yy                  # sum of y
    k_m11 = xx * yy             # sum of x*y
    k_m20 = xx * xx             # sum of x^2
    k_m02 = yy * yy             # sum of y^2

    kernels = torch.stack([k_m00, k_m10, k_m01, k_m11, k_m20, k_m02], dim=0)
    kernels = kernels.unsqueeze(1)  # (6, 1, W, W)
    return kernels, half_w


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def find_gaussians_torch(
    tensor: torch.Tensor,
    adaptive_gauss_fit: bool = False,
    plot_heatmaps: bool = False,
    plotter=None,
    log_missing_gaussians: bool = True,
    missing_gaussians_logger=None,
    adaptive_threshold: float = 0.003,
    adaptive_n_sigma: float = 3.0,
    adaptive_max_iter: int = 10,
    adaptive_min_half_w: int = 1,
    adaptive_max_half_w: int = 5,
    fixed_threshold: float = 0.008,
    fixed_window_size: int = 4,
    device: str | torch.device = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized torch counterpart of :func:`find_gaussians`.

    Parameters mirror the numpy version.  ``adaptive_gauss_fit=True`` is *not*
    implemented in this batched path and silently falls back to the fixed
    window variant (equivalent to ``adaptive_gauss_fit=False`` in the numpy
    reference).  See the docstring for details.

    Parameters
    ----------
    tensor : torch.Tensor
        Logits of shape ``(out_patch_dim**2, in_patch_dim, in_patch_dim)``.
    adaptive_gauss_fit : bool
        Kept for API symmetry.  When ``True`` we still run the fixed-window
        variant -- the adaptive per-peak window iteration is hard to vectorise
        without a Python loop over peaks.  Documented as an algorithmic
        compromise for this batched implementation.
    device : str or torch.device
        Where to run the heavy compute.  Default ``"cpu"``.
    """
    if plot_heatmaps and plotter is not None:
        # Plotting is interactive and per-patch; not supported in the vectorized
        # path.  Mirror numpy behaviour silently for callers that don't actually
        # use it.
        raise NotImplementedError(
            "plot_heatmaps is not supported in find_gaussians_torch."
        )

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("`tensor` must be a torch.Tensor.")

    M = tensor.shape[0]
    out_patch_size = int(M ** 0.5)
    assert out_patch_size * out_patch_size == M, (
        "Tensor's first dimension must be a perfect square."
    )
    in_h, in_w = int(tensor.shape[1]), int(tensor.shape[2])

    target_device = torch.device(device)
    work_dtype = torch.float32

    logits = tensor.to(device=target_device, dtype=work_dtype, non_blocking=True)

    # Step 1: Softmax and reshape to heatmaps
    heatmaps = _softmax_reshape_heatmaps(logits, out_patch_size, in_h, in_w)

    # Step 2: Compute global peak per patch (pre-blur)
    global_peaks_xy = _compute_global_peaks(heatmaps, out_patch_size, work_dtype)

    # Step 3: Apply blur and detect local maxima
    blurred, peaks_mask = _apply_blur_and_detect_peaks(
        heatmaps, fixed_threshold, fixed_window_size, target_device, work_dtype
    )

    # Check if no peaks were found
    if not bool(peaks_mask.any()):
        empty = np.zeros((0, 2), dtype=np.float32)
        empty_cov = np.zeros((0, 2, 2), dtype=np.float32)
        if log_missing_gaussians:
            for py in range(in_h):
                for px in range(in_w):
                    print(f"No Gaussians found in patch ({px}, {py})")
        if missing_gaussians_logger is not None:
            for py in range(in_h):
                for px in range(in_w):
                    missing_gaussians_logger(px, py)
        return empty, empty, empty, empty_cov

    # Step 4: Gather windows and compute moments
    (
        patch_idx, py_peak, px_peak,
        local_mean_x, local_mean_y,
        cov_xx, cov_yy, cov_xy,
        x_min_clamped, y_min_clamped,
    ) = _gather_and_compute_moments(
        blurred, peaks_mask, fixed_window_size, out_patch_size,
        target_device, work_dtype
    )

    if patch_idx.numel() == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        empty_cov = np.zeros((0, 2, 2), dtype=np.float32)
        return empty, empty, empty, empty_cov

    # Step 5: Assemble correspondence arrays
    pts_A, means_B, peaks_B, covs_B = _assemble_torch_correspondences(
        patch_idx, py_peak, px_peak,
        local_mean_x, local_mean_y,
        cov_xx, cov_yy, cov_xy,
        x_min_clamped, y_min_clamped,
        global_peaks_xy,
        in_h, in_w, work_dtype
    )

    # Optional missing-patch logging
    if log_missing_gaussians or missing_gaussians_logger is not None:
        any_peak_per_patch = peaks_mask.view(-1, out_patch_size * out_patch_size).any(dim=1).cpu().numpy()
        if not any_peak_per_patch.all():
            for flat_i, has in enumerate(any_peak_per_patch):
                if has:
                    continue
                py = flat_i // in_w
                px = flat_i % in_w
                if log_missing_gaussians:
                    print(f"No Gaussians found in patch ({px}, {py})")
                if missing_gaussians_logger is not None:
                    missing_gaussians_logger(px, py)

    # Convert to numpy
    pts_A_np = pts_A.detach().cpu().numpy().astype(np.float32, copy=False)
    means_B_np = means_B.detach().cpu().numpy().astype(np.float32, copy=False)
    peaks_B_np = peaks_B.detach().cpu().numpy().astype(np.float32, copy=False)
    covs_B_np = covs_B.detach().cpu().numpy().astype(np.float32, copy=False)

    return pts_A_np, means_B_np, peaks_B_np, covs_B_np



# --------------------------------------------------------------------------- #
# Batched extractor                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class GaussiansBatchTensors:
    """GPU-resident batched extractor output.

    All tensors stay on the same device as the input ``logits`` and share
    the same first dim ``N_total`` (sum of per-frame correspondence counts).
    ``batch_idx[k]`` says which frame peak ``k`` belongs to (sorted, so per-
    frame slices are contiguous).

    Use :func:`pad_for_batched_lm_from_gpu` (in homography_torch_lm) to pack
    these into the (B, N_max, ...) padded tensors needed by the batched LM —
    no host round-trip.
    """
    pts_A: torch.Tensor       # (N_total, 2) float32
    means_B: torch.Tensor     # (N_total, 2) float32
    peaks_B: torch.Tensor     # (N_total, 2) float32
    covs_B: torch.Tensor      # (N_total, 2, 2) float32
    batch_idx: torch.Tensor   # (N_total,)    int64 — frame index for each peak
    counts: torch.Tensor      # (B,)          int64 — how many peaks per frame
    B: int                    # batch size (for downstream sanity checks)
    # softmax mass at the post-blur peak cell — usable as per-mode confidence
    # weight in RANSAC sampling. Legacy callers ignore it.
    peak_values: torch.Tensor = None  # (N_total,)  float32  ∈ [0, 1]


def find_gaussians_torch_batch(
    tensor: torch.Tensor,
    fixed_threshold: float = 0.008,
    fixed_window_size: int = 4,
    device: str | torch.device = "cpu",
    return_tensors: bool = False,
):
    """Batched version of :func:`find_gaussians_torch`.

    Processes a stack of B logit tensors in ONE batched torch pass (softmax,
    blur, max-pool, peak-window gather, moments — all done once across
    ``B * in_h * in_w`` patches).

    Parameters
    ----------
    tensor : torch.Tensor
        Logits of shape ``(B, M, in_h, in_w)`` where ``M = out_patch_dim**2``.
    fixed_threshold, fixed_window_size, device
        Same meaning as in :func:`find_gaussians_torch`. The batched path
        only supports the fixed-window variant — ``adaptive_gauss_fit`` is
        not exposed because per-peak adaptive iteration cannot be vectorized
        cleanly.
    return_tensors : bool
        If ``False`` (default, kept for backward compat), return a Python
        list of B ``(pts_A, means_B, peaks_B, covs_B)`` numpy tuples — this
        forces a GPU→CPU transfer of every per-frame array.

        If ``True``, return a :class:`GaussiansBatchTensors` whose tensors
        stay on ``device``. Use this when the downstream consumer (e.g.
        :func:`pad_for_batched_lm_from_gpu` + the batched torch-LM) is also
        on GPU — eliminates the GPU→CPU→GPU round-trip.

    Notes
    -----
    The win over ``[find_gaussians_torch(t) for t in batch]`` is amortizing
    per-frame Python overhead and kernel launches. On CUDA the difference is
    biggest at small batch sizes where launch latency dominates.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("`tensor` must be a torch.Tensor.")
    if tensor.dim() != 4:
        raise ValueError(
            f"Batched tensor must be (B, M, in_h, in_w), got {tuple(tensor.shape)}"
        )

    B, M, in_h, in_w = tensor.shape
    out_patch_size = int(M ** 0.5)
    if out_patch_size * out_patch_size != M:
        raise ValueError(
            "Tensor's second dimension (M) must be a perfect square."
        )
    n_patches_per_frame = in_h * in_w
    n_patches_total = B * n_patches_per_frame

    target_device = torch.device(device)
    work_dtype = torch.float32

    logits = tensor.to(device=target_device, dtype=work_dtype, non_blocking=True)

    # 1. Softmax over M axis, per (b, py, px).
    soft = torch.softmax(logits, dim=1)                          # (B, M, in_h, in_w)
    # Move M to last so we can reshape to (B*n_patches, O, O).
    soft = soft.permute(0, 2, 3, 1).contiguous()                 # (B, in_h, in_w, M)
    heatmaps = soft.view(n_patches_total, out_patch_size, out_patch_size)

    # 2. Global peak per patch (pre-blur).
    flat = heatmaps.reshape(n_patches_total, -1)
    global_peak_idx = torch.argmax(flat, dim=1)
    global_peak_y = (global_peak_idx // out_patch_size).to(work_dtype)
    global_peak_x = (global_peak_idx %  out_patch_size).to(work_dtype)
    global_peaks_xy = torch.stack([global_peak_x, global_peak_y], dim=1)

    # 3. Gaussian blur (cv2-equivalent 5x5 sigma=1.1, replicate-pad).
    blur_kernel = _cv2_gaussian_kernel_5x5(
        sigma=1.1, device=target_device, dtype=work_dtype
    ).view(1, 1, 5, 5)
    heatmaps_padded = F.pad(heatmaps.unsqueeze(1), (2, 2, 2, 2), mode="replicate")
    blurred = F.conv2d(heatmaps_padded, blur_kernel, padding=0)  # (P, 1, O, O)

    # 4. Local maxima via max_pool2d.
    W = int(fixed_window_size)
    pad_max = W // 2
    pooled = F.max_pool2d(blurred, kernel_size=W, stride=1, padding=pad_max)
    if pooled.shape[-1] != blurred.shape[-1]:
        pooled = pooled[..., :blurred.shape[-2], :blurred.shape[-1]]
    peaks_mask = (blurred > fixed_threshold) & (blurred == pooled)
    peaks_mask = peaks_mask.squeeze(1)                            # (P, O, O)

    # 5. Locate peaks, gather per-peak windows, compute moments.
    half_w = W // 2
    full_window = 2 * half_w + 1
    blurred_padded_full = F.pad(blurred, (half_w, half_w, half_w, half_w))

    coords_o = torch.arange(out_patch_size, device=target_device, dtype=work_dtype)
    grid_y_o, grid_x_o = torch.meshgrid(coords_o, coords_o, indexing="ij")
    x_min_clamped = (grid_x_o - half_w).clamp(min=0)
    y_min_clamped = (grid_y_o - half_w).clamp(min=0)

    # Always-list outputs path: even with zero peaks across the whole batch
    # we must return B empty tuples, not bail out.
    empty_pts = np.zeros((0, 2), dtype=np.float32)
    empty_cov = np.zeros((0, 2, 2), dtype=np.float32)

    def _empty_tensor_result() -> GaussiansBatchTensors:
        z2 = torch.zeros((0, 2), dtype=work_dtype, device=target_device)
        z22 = torch.zeros((0, 2, 2), dtype=work_dtype, device=target_device)
        z1 = torch.zeros((0,), dtype=work_dtype, device=target_device)
        return GaussiansBatchTensors(
            pts_A=z2, means_B=z2.clone(), peaks_B=z2.clone(), covs_B=z22,
            batch_idx=torch.zeros((0,), dtype=torch.long, device=target_device),
            counts=torch.zeros((B,), dtype=torch.long, device=target_device),
            B=B, peak_values=z1,
        )

    if not bool(peaks_mask.any()):
        if return_tensors:
            return _empty_tensor_result()
        return [
            (empty_pts.copy(), empty_pts.copy(), empty_pts.copy(), empty_cov.copy())
            for _ in range(B)
        ]

    patch_idx, py_peak, px_peak = peaks_mask.nonzero(as_tuple=True)

    dy = torch.arange(full_window, device=target_device)
    dx = torch.arange(full_window, device=target_device)
    dy_grid, dx_grid = torch.meshgrid(dy, dx, indexing="ij")
    py_idx = py_peak[:, None, None] + dy_grid[None]
    px_idx = px_peak[:, None, None] + dx_grid[None]
    windows = blurred_padded_full[patch_idx[:, None, None], 0, py_idx, px_idx]

    coord = torch.arange(full_window, device=target_device, dtype=work_dtype)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    m00 = windows.sum(dim=(-1, -2))
    m10 = (windows * xx).sum(dim=(-1, -2))
    m01 = (windows * yy).sum(dim=(-1, -2))
    m11 = (windows * xx * yy).sum(dim=(-1, -2))
    m20 = (windows * xx * xx).sum(dim=(-1, -2))
    m02 = (windows * yy * yy).sum(dim=(-1, -2))

    keep = m00 > 0
    if not bool(keep.all()):
        patch_idx = patch_idx[keep]
        py_peak = py_peak[keep]
        px_peak = px_peak[keep]
        m00 = m00[keep]; m10 = m10[keep]; m01 = m01[keep]
        m11 = m11[keep]; m20 = m20[keep]; m02 = m02[keep]

    if patch_idx.numel() == 0:
        if return_tensors:
            return _empty_tensor_result()
        return [
            (empty_pts.copy(), empty_pts.copy(), empty_pts.copy(), empty_cov.copy())
            for _ in range(B)
        ]

    local_mean_x = m10 / m00
    local_mean_y = m01 / m00
    cov_xx = m20 / m00 - local_mean_x * local_mean_x
    cov_yy = m02 / m00 - local_mean_y * local_mean_y
    cov_xy = m11 / m00 - local_mean_x * local_mean_y

    sel_mean_x = x_min_clamped[py_peak, px_peak] + local_mean_x
    sel_mean_y = y_min_clamped[py_peak, px_peak] + local_mean_y
    sel_cov_xx = cov_xx + 1e-4
    sel_cov_yy = cov_yy + 1e-4
    sel_cov_xy = cov_xy

    # Per-peak (b, py_A, px_A): patch_idx is into (B * n_patches_per_frame).
    batch_idx = patch_idx // n_patches_per_frame
    in_frame_patch = patch_idx % n_patches_per_frame
    patch_py_A = (in_frame_patch // in_w).to(work_dtype)
    patch_px_A = (in_frame_patch % in_w).to(work_dtype)

    pts_A_all = torch.stack([patch_px_A, patch_py_A], dim=1)        # (N_total, 2)
    means_B_all = torch.stack([sel_mean_x, sel_mean_y], dim=1)
    peaks_B_all = global_peaks_xy[patch_idx]
    covs_B_all = torch.stack(
        [
            torch.stack([sel_cov_xx, sel_cov_xy], dim=1),
            torch.stack([sel_cov_xy, sel_cov_yy], dim=1),
        ],
        dim=1,
    )                                                                # (N_total, 2, 2)

    # Per-peak softmax mass at the post-blur peak cell — confidence weight for
    # weighted RANSAC sampling (pipeline.py ransac_sampling kwarg).
    peak_values_all = blurred.squeeze(1)[patch_idx, py_peak, px_peak]

    if return_tensors:
        # Per-frame counts for downstream padding. bincount needs a contiguous
        # 0..B-1 range; we know batch_idx is sorted (see comment below) so a
        # single GPU op suffices.
        counts = torch.bincount(batch_idx, minlength=B)              # (B,)
        return GaussiansBatchTensors(
            pts_A=pts_A_all, means_B=means_B_all,
            peaks_B=peaks_B_all, covs_B=covs_B_all,
            batch_idx=batch_idx, counts=counts, B=B,
            peak_values=peak_values_all,
        )

    # One bulk transfer to host, then per-frame slicing in numpy (cheap).
    batch_idx_np = batch_idx.detach().cpu().numpy()
    pts_A_np = pts_A_all.detach().cpu().numpy().astype(np.float32, copy=False)
    means_B_np = means_B_all.detach().cpu().numpy().astype(np.float32, copy=False)
    peaks_B_np = peaks_B_all.detach().cpu().numpy().astype(np.float32, copy=False)
    covs_B_np = covs_B_all.detach().cpu().numpy().astype(np.float32, copy=False)

    out: list[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    # Use np.searchsorted to slice efficiently — batch_idx is already sorted
    # because nonzero() walks (patch_idx, ...) in row-major order and patch
    # indices for frame b come before frame b+1.
    if batch_idx_np.size > 0 and not np.all(np.diff(batch_idx_np) >= 0):
        # Defensive fallback: sort once if the assumption ever breaks.
        order = np.argsort(batch_idx_np, kind="stable")
        batch_idx_np = batch_idx_np[order]
        pts_A_np = pts_A_np[order]
        means_B_np = means_B_np[order]
        peaks_B_np = peaks_B_np[order]
        covs_B_np = covs_B_np[order]

    starts = np.searchsorted(batch_idx_np, np.arange(B), side="left")
    stops = np.searchsorted(batch_idx_np, np.arange(B), side="right")
    for b in range(B):
        s, e = int(starts[b]), int(stops[b])
        if s == e:
            out.append(
                (empty_pts.copy(), empty_pts.copy(), empty_pts.copy(), empty_cov.copy())
            )
        else:
            out.append((pts_A_np[s:e], means_B_np[s:e], peaks_B_np[s:e], covs_B_np[s:e]))
    return out
