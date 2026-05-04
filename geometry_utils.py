"""Shared geometric operations and transformations.

Eliminates duplication of geometry primitives across numpy and torch backends.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# NumPy Backend
# ============================================================================

def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Vectorized projection of Nx2 points using a 3x3 homography matrix.

    Args:
        H: (3, 3) homography matrix
        pts: (N, 2) points in source space

    Returns:
        (N, 2) projected points in target space
    """
    pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    proj = (H @ pts_homo.T).T
    z = proj[:, 2:] + 1e-6
    proj_2d = proj[:, :2] / z
    return proj_2d


def srt_to_matrix(params: np.ndarray) -> np.ndarray:
    """
    Build a 3x3 similarity (sRT) matrix from [s, theta, tx, ty].

    Args:
        params: (4,) array [scale, theta (radians), tx, ty]

    Returns:
        (3, 3) similarity transformation matrix
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


def extract_srt_from_homography(H: np.ndarray) -> np.ndarray:
    """
    Extract the best-fit sRT parameters [s, theta, tx, ty] from a general
    3x3 homography by reading the top-left 2x2 block.

    Args:
        H: (3, 3) homography matrix

    Returns:
        (4,) array [s, theta, tx, ty]
    """
    a = H[0, 0]
    b = H[1, 0]
    tx = H[0, 2]
    ty = H[1, 2]
    s = np.sqrt(a**2 + b**2)
    theta = np.arctan2(b, a)
    return np.array([s, theta, tx, ty])


def compute_corner_error(
    H_gt: np.ndarray,
    H_pred: np.ndarray,
    w: float = 1024,
    h: float = 1024,
) -> float:
    """
    Compute mean L2 corner error between two homographies.

    Args:
        H_gt: Ground truth (3, 3) homography
        H_pred: Predicted (3, 3) homography
        w: Image width
        h: Image height

    Returns:
        Mean L2 distance of projected corners
    """
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    corners_gt = project_points(H_gt, corners)
    corners_pred = project_points(H_pred, corners)
    error = np.linalg.norm(corners_gt - corners_pred, axis=1).mean()
    return float(error)


# ============================================================================
# Torch Backend
# ============================================================================

if HAS_TORCH:

    def project_points_torch(H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """
        Torch version of project_points.

        Args:
            H: (3, 3) or (B, 3, 3) homography matrix/matrices
            pts: (N, 2) or (B, N, 2) points

        Returns:
            (N, 2) or (B, N, 2) projected points
        """
        ones = torch.ones((pts.shape[-2], 1), dtype=pts.dtype, device=pts.device)
        if pts.dim() == 3:
            ones = ones.unsqueeze(0).expand(pts.shape[0], -1, -1)
        pts_homo = torch.cat([pts, ones], dim=-1)
        proj = (H @ pts_homo.transpose(-2, -1)).transpose(-2, -1)
        z = proj[..., 2:3] + 1e-6
        proj_2d = proj[..., :2] / z
        return proj_2d

    def srt_to_matrix_torch(params: torch.Tensor) -> torch.Tensor:
        """
        Torch version of srt_to_matrix.

        Args:
            params: (4,) or (B, 4) tensor [s, theta, tx, ty]

        Returns:
            (3, 3) or (B, 3, 3) similarity matrix/matrices
        """
        if params.dim() == 1:
            s, theta, tx, ty = params[0], params[1], params[2], params[3]
            s = torch.clamp(s, 0.0, 4.0)
            theta = torch.clamp(
                theta,
                float(np.radians(-90.0)),
                float(np.radians(90.0)),
            )
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
        else:
            s, theta, tx, ty = (
                params[:, 0],
                params[:, 1],
                params[:, 2],
                params[:, 3],
            )
            s = torch.clamp(s, 0.0, 4.0)
            theta = torch.clamp(
                theta,
                float(np.radians(-90.0)),
                float(np.radians(90.0)),
            )
            c, si = torch.cos(theta), torch.sin(theta)
            B = params.shape[0]
            result = torch.zeros((B, 3, 3), dtype=params.dtype, device=params.device)
            result[:, 0, 0] = s * c
            result[:, 0, 1] = -s * si
            result[:, 0, 2] = tx
            result[:, 1, 0] = s * si
            result[:, 1, 1] = s * c
            result[:, 1, 2] = ty
            result[:, 2, 2] = 1.0
            return result

    def extract_srt_from_homography_torch(H: torch.Tensor) -> torch.Tensor:
        """
        Torch version of extract_srt_from_homography.

        Args:
            H: (3, 3) or (B, 3, 3) homography matrix/matrices

        Returns:
            (4,) or (B, 4) array [s, theta, tx, ty]
        """
        if H.dim() == 2:
            a = H[0, 0]
            b = H[1, 0]
            tx = H[0, 2]
            ty = H[1, 2]
        else:
            a = H[:, 0, 0]
            b = H[:, 1, 0]
            tx = H[:, 0, 2]
            ty = H[:, 1, 2]

        s = torch.sqrt(a**2 + b**2)
        theta = torch.atan2(b, a)

        if H.dim() == 2:
            return torch.stack([s, theta, tx, ty])
        else:
            return torch.stack([s, theta, tx, ty], dim=1)

    def compute_corner_error_torch(
        H_gt: torch.Tensor,
        H_pred: torch.Tensor,
        w: float = 1024,
        h: float = 1024,
    ) -> torch.Tensor:
        """
        Torch version of compute_corner_error.

        Args:
            H_gt: (3, 3) or (B, 3, 3) ground truth homography
            H_pred: (3, 3) or (B, 3, 3) predicted homography
            w: Image width
            h: Image height

        Returns:
            Scalar or (B,) mean corner error
        """
        corners = torch.tensor(
            [[0, 0], [w, 0], [w, h], [0, h]],
            dtype=H_gt.dtype,
            device=H_gt.device,
        )
        if H_gt.dim() == 3:
            corners = corners.unsqueeze(0).expand(H_gt.shape[0], -1, -1)

        corners_gt = project_points_torch(H_gt, corners)
        corners_pred = project_points_torch(H_pred, corners)
        error = torch.norm(corners_gt - corners_pred, dim=-1).mean(dim=-1)
        return error


# ============================================================================
# Unified Interface (Backend Selection)
# ============================================================================

def project_points_unified(
    H: Union[np.ndarray, "torch.Tensor"],
    pts: Union[np.ndarray, "torch.Tensor"],
) -> Union[np.ndarray, "torch.Tensor"]:
    """Backend-agnostic point projection."""
    if HAS_TORCH and isinstance(H, torch.Tensor):
        return project_points_torch(H, pts)
    return project_points(H, pts)


def srt_to_matrix_unified(
    params: Union[np.ndarray, "torch.Tensor"],
) -> Union[np.ndarray, "torch.Tensor"]:
    """Backend-agnostic sRT matrix construction."""
    if HAS_TORCH and isinstance(params, torch.Tensor):
        return srt_to_matrix_torch(params)
    return srt_to_matrix(params)


def extract_srt_from_homography_unified(
    H: Union[np.ndarray, "torch.Tensor"],
) -> Union[np.ndarray, "torch.Tensor"]:
    """Backend-agnostic sRT extraction."""
    if HAS_TORCH and isinstance(H, torch.Tensor):
        return extract_srt_from_homography_torch(H)
    return extract_srt_from_homography(H)
