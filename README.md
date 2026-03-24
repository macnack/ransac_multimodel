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
- `ransac_multimodel/homography_torch.py`
  - `project_points_torch`
  - `homography_residuals_vectorized_torch`
  - `srt_residuals_torch`
  - `optimize_homography_torch`
- `ransac_multimodel/parity_utils.py`
  - deterministic seed helpers
  - NumPy <-> Torch conversion helpers
  - JSON utility helpers for benchmark outputs
- `ransac_multimodel/transforms.py`
  - `convert_to_pixel_homography`
  - `convert_to_dataloader_homography`
- `ransac_multimodel/plotting.py`
  - plotting utilities
- `benchmarks/benchmark_numpy_vs_torch.py`
  - residual-only and end-to-end benchmarks
  - CPU baseline + Torch CPU + optional Torch CUDA

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

## Benchmark

Entry point:

```bash
python -m benchmarks.benchmark_numpy_vs_torch --modes all --device auto
```

Useful options:

```bash
# synthetic fast smoke benchmark
python -m benchmarks.benchmark_numpy_vs_torch --synthetic --synthetic-n 20000 --modes residual --repeats 5 --warmup 2 --device auto --quiet

# run on selected tensor samples
python -m benchmarks.benchmark_numpy_vs_torch --sample-ids 98,122,128 --modes all --model sRT --device auto --quiet
```

Output:
- console summary with median and p95 latencies
- JSON result file at `benchmarks/results/latest.json`
- timestamped JSON snapshot in `benchmarks/results/`

CUDA behavior:
- `--device auto`: runs CPU benchmarks always, and runs CUDA benchmarks only when CUDA is available
- when CUDA is unavailable, GPU runs are skipped with explicit notice

Quiet behavior:
- `--quiet` suppresses noisy per-iteration logging from Gaussian extraction and sRT optimizer prints

Synthetic scaling:
- use `--synthetic-n <N>` to increase correspondence count (for example `20000`) so CUDA has a fairer workload

Parity notes:
- residual parity uses strict `np.allclose(..., rtol=1e-6, atol=1e-7)` checks
- end-to-end parity reports homography Frobenius difference, absolute corner errors, and corner-error delta

## SAT-ROMA Ablation Runner

Use the new experiment runner when you want reproducible baseline vs grid-search tuning and manual review outputs from only a dataset folder.

Entry point:

```bash
.venv/bin/python experiments/sat_roma_ablation.py --dataset-dir ./tensors --mode both --write-review-images
```

Key options:

```bash
# baseline only (preserves baseline behavior path)
.venv/bin/python experiments/sat_roma_ablation.py --dataset-dir ./tensors --mode baseline

# coarse grid only, explicit samples, deterministic seed
.venv/bin/python experiments/sat_roma_ablation.py --dataset-dir ./tensors --sample-ids 98,122,128 --mode grid --max-grid-configs 64 --seed 1234 --write-review-images

# with split file and optional fine search around best coarse config
.venv/bin/python experiments/sat_roma_ablation.py --dataset-dir ./tensors --split-file ./splits/val_ids.txt --mode both --run-fine-search --seed 1234

# Optuna search (TPE sampler) instead of coarse grid
.venv/bin/python experiments/sat_roma_ablation.py --dataset-dir ./tensors --mode optuna --optuna-trials 40 --optuna-startup-trials 10 --seed 1234
```

Outputs are saved under `experiments/sat_roma_runs/<run_name>/`:
- `per_sample_metrics.csv` and `per_sample_metrics.json`
- `configs_ranked.csv` and `configs_ranked.json` ranked by corner error (primary) then runtime (secondary)
- `review_index.csv` and `review_index.json` with status labels (`KEEP_FOR_OPTIMIZATION` or `EXCLUDE`) and reasons
- `optimize_on.txt` and `do_not_use_for_tuning.txt`
- `summary_report.md` with baseline vs best config summary
- when `--mode optuna`: `optuna_trials.csv` and `optuna_trials.json` with trial params, values, and linked `config_id`

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
