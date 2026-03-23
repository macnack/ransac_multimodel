# Agent Context

## Goal
This repository was refactored so the reusable code lives inside `ransac_multimodel/` as a copy-paste Python 3 submodule.

## Architecture
- `ransac_multimodel/gaussian_fit.py`: Gaussian extraction from heatmaps.
- `ransac_multimodel/correspondence.py`: Converts tensor heatmaps into correspondence arrays (`pts_A`, `means_B`, `peaks_B`, `covs_B`).
- `ransac_multimodel/homography.py`: RANSAC init + least-squares optimization (`full` and `sRT`) and corner error.
- `ransac_multimodel/transforms.py`: feature-grid homography -> pixel and dataloader-space homography conversions.
- `ransac_multimodel/plotting.py`: plotting and visualization tools.

## Behavioral constraints
- Preserve existing math and constants unless explicitly asked to change.
- Do not change thresholds, bounds, regularization weights, or scaling constants by default.
- Prefer refactor/organization changes over algorithmic changes.

## Dependencies
- Required: `numpy`, `scipy`, `opencv-python`, `matplotlib`, `torch`.

## Known notes
- `solve.py` is an example runner and uses local sample `.pt` files.
- The package `__init__.py` is intentionally minimal to avoid eager heavy imports.
- If making future improvements, keep backward-compatible function signatures where possible.

## Suggested next tasks
1. Add automated tests for `project_points`, homography conversions, and shape/type checks.
2. Add `pyproject.toml` for packaging/install (`pip install -e .`).