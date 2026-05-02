"""Reusable helpers for Gaussian correspondence extraction and homography fitting.

Most callers want the high-level pipeline API:

    from ransac_multimodel import (
        estimate_homography, estimate_homography_batched,
        BatchedHomographyResult, HomographyResult,
        BACKENDS, resolve_backend,
        RANSAC_METHODS, resolve_ransac_method,
    )

The pipeline module exposes the rest (LMHistory etc.) via attribute lookup;
the imports above cover the CLI-friendly surface.
"""

from .pipeline import (  # noqa: F401
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_BATCHED_BACKEND,
    DEFAULT_RANSAC_METHOD_NAME,
    BatchedHomographyResult,
    HomographyResult,
    RANSAC_METHODS,
    estimate_homography,
    estimate_homography_batched,
    resolve_backend,
    resolve_ransac_method,
)
