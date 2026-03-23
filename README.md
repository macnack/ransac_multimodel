# ransac_multimodel

Reusable Python 3 submodule for:
- extracting Gaussian correspondences from heatmaps,
- robust homography fitting (full or sRT),
- homography coordinate-space conversions,
- visualization helpers.

The code was factored from `solve.py` without changing core constants/behavior in the optimization and Gaussian extraction logic.

## Module structure

- `ransac_multimodel/gaussian_fit.py`
  - `extract_gaussians_adaptive`
  - `extract_gaussians_from_heatmap2`
- `ransac_multimodel/correspondence.py`
  - `find_gaussians`
- `ransac_multimodel/homography.py`
  - `project_points`
  - `optimize_homography`
  - `compute_corner_error`
- `ransac_multimodel/transforms.py`
  - `convert_to_pixel_homography`
  - `convert_to_dataloader_homography`
- `ransac_multimodel/plotting.py`
  - plotting utilities

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick usage

```python
from ransac_multimodel.correspondence import find_gaussians
from ransac_multimodel.homography import optimize_homography, compute_corner_error
from ransac_multimodel.transforms import (
    convert_to_dataloader_homography,
    convert_to_pixel_homography,
)

# pts_A, means_B, peaks_B, covs_B = find_gaussians(logits)
# H_final, H_init = optimize_homography(pts_A, means_B, covs_B, peaks_B=peaks_B, model='sRT')
```

## Copy to a new project

Copy only this folder:

- `ransac_multimodel/`

Optionally also copy:

- `requirements.txt`
- `AGENT_CONTEXT.md` (if another LLM/engineer will continue work)

## What can be removed in this repo (if you only need the submodule)

Safe to remove for a minimal library-only package:
- sample assets/data: `*.pt`, `*.png`
- scratch scripts: `check_images.py`, `check_tensors.py`, `solve.py`
- legacy duplicate modules: `fit_gaus_old.py`, `plot_utils.py`
- cache folders: `__pycache__/`

Keep:
- `ransac_multimodel/`
- `requirements.txt`
- this `README.md`
- `AGENT_CONTEXT.md` (optional but recommended)

## Notes

- Runtime dependencies like OpenCV (`cv2`) and PyTorch (`torch`) are required for full pipeline execution.
- Plotting functions require Matplotlib.
