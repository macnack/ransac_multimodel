#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ransac_multimodel.correspondence import find_gaussians
from ransac_multimodel.homography import compute_corner_error, optimize_homography
from ransac_multimodel.parity_utils import set_deterministic_seeds
from ransac_multimodel.transforms import (
    convert_to_dataloader_homography,
    convert_to_pixel_homography,
)


@dataclass
class SampleData:
    sample_id: int
    logits: Any
    H_gt: np.ndarray
    im_A: Any
    im_B: Any


@dataclass
class TriageThresholds:
    min_correspondences: int = 8
    min_inliers: int = 4
    min_inlier_ratio: float = 0.05
    max_h_cond: float = 1e6
    max_corner_error_fixed: float = 3000.0
    outlier_iqr_k: float = 2.5


def _tensor_to_numpy_image(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)
    except Exception:
        arr = np.asarray(x)

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    arr = (arr - mn) / (mx - mn + 1e-8)
    return np.clip(arr, 0.0, 1.0)


def _to_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_mean(values):
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _safe_median(values):
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.median(vals))


def parse_sample_ids(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    if "," in raw:
        out = []
        for tok in raw.split(","):
            tok = tok.strip()
            if tok:
                out.append(int(tok))
        return sorted(set(out))
    if "-" in raw:
        lo_s, hi_s = raw.split("-", maxsplit=1)
        lo, hi = int(lo_s), int(hi_s)
        if lo > hi:
            raise ValueError("Invalid sample range: start > end")
        return list(range(lo, hi + 1))
    return [int(raw)]


def discover_sample_ids(dataset_dir: Path) -> list[int]:
    input_ids = set()
    for path in dataset_dir.glob("input_sample_*.pt"):
        m = re.search(r"input_sample_(\d+)\.pt$", path.name)
        if m:
            input_ids.add(int(m.group(1)))

    sample_ids = set()
    for path in dataset_dir.glob("sample_*_tensor.pt"):
        m = re.search(r"sample_(\d+)_tensor\.pt$", path.name)
        if m:
            sample_ids.add(int(m.group(1)))

    ids = sorted(input_ids.intersection(sample_ids))
    return ids


def load_ids_from_split_file(split_file: Path) -> list[int]:
    text = split_file.read_text(encoding="utf-8")
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            for tok in line.split(","):
                tok = tok.strip()
                if tok:
                    ids.append(int(tok))
        else:
            ids.append(int(line))
    return sorted(set(ids))


def load_sample(dataset_dir: Path, sample_id: int) -> SampleData:
    import torch

    input_path = dataset_dir / f"input_sample_{sample_id:06d}.pt"
    sample_path = dataset_dir / f"sample_{sample_id:03d}_tensor.pt"

    gt = torch.load(input_path, map_location=torch.device("cpu"))
    sample = torch.load(sample_path, map_location=torch.device("cpu"))

    logits = sample[16]["gm_cls"][0]
    H_gt = gt["homography_gt"]
    H_gt = H_gt.numpy() if hasattr(H_gt, "numpy") else np.asarray(H_gt)

    return SampleData(
        sample_id=sample_id,
        logits=logits,
        H_gt=np.asarray(H_gt, dtype=np.float64),
        im_A=gt["im_A"],
        im_B=gt["im_B"],
    )


def default_config() -> dict[str, Any]:
    return {
        "adaptive_gauss_fit": True,
        "adaptive_threshold": 0.003,
        "adaptive_n_sigma": 3.0,
        "adaptive_max_iter": 10,
        "adaptive_min_half_w": 1,
        "adaptive_max_half_w": 5,
        "fixed_threshold": 0.008,
        "fixed_window_size": 4,
        "use_means_for_ransac": False,
        "model": "sRT",
        "ransac_method": "RANSAC",
        "ransac_reproj_threshold": 3.0,
        "ransac_max_iters": 5000,
        "ransac_confidence": 0.995,
        "robust_loss": "huber",
        "f_scale": 2.0,
        "max_nfev": 5000,
        "srt_x_scale": [0.20, 1.0, 1.0, 1.0],
        "srt_bounds_profile": "default",
        "srt_scale_reg_weight": 0.01,
        "srt_rot_reg_weight": 0.001,
    }


def resolve_srt_bounds(profile: str):
    if profile == "tight":
        return ([0.50, np.radians(-90), -30, -30], [2.0, np.radians(90), 30, 30])
    return ([0.25, np.radians(-180), -60, -60], [3.0, np.radians(180), 60, 60])


def create_coarse_grid(seed: int, max_configs: int) -> list[dict[str, Any]]:
    base = default_config()

    base_grid = {
        "adaptive_gauss_fit": [True, False],
        "use_means_for_ransac": [False, True],
        "model": ["sRT", "full"],
        "ransac_reproj_threshold": [2.0, 3.0],
        "ransac_max_iters": [2000, 5000],
        "robust_loss": ["huber", "soft_l1"],
        "f_scale": [1.0, 2.0],
        "srt_bounds_profile": ["default", "tight"],
        "srt_scale_reg_weight": [0.005, 0.01],
        "srt_rot_reg_weight": [0.0005, 0.001],
    }

    keys = list(base_grid.keys())
    combos = []
    for values in itertools.product(*[base_grid[k] for k in keys]):
        cfg = dict(base)
        for k, v in zip(keys, values):
            cfg[k] = v

        if cfg["model"] == "full":
            cfg["srt_bounds_profile"] = "default"
            cfg["srt_scale_reg_weight"] = base["srt_scale_reg_weight"]
            cfg["srt_rot_reg_weight"] = base["srt_rot_reg_weight"]

        combos.append(cfg)

    unique = []
    seen = set()
    for cfg in combos:
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(cfg)

    rng = np.random.default_rng(seed)
    order = np.arange(len(unique))
    rng.shuffle(order)
    unique = [unique[i] for i in order]

    baseline = base
    unique = [baseline] + [u for u in unique if json.dumps(u, sort_keys=True) != json.dumps(baseline, sort_keys=True)]

    if max_configs > 0:
        unique = unique[:max_configs]
    return unique


def suggest_optuna_config(trial, base: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(base)

    cfg["adaptive_gauss_fit"] = trial.suggest_categorical("adaptive_gauss_fit", [True, False])
    cfg["use_means_for_ransac"] = trial.suggest_categorical("use_means_for_ransac", [False, True])
    cfg["model"] = trial.suggest_categorical("model", ["sRT", "full"])

    cfg["ransac_reproj_threshold"] = trial.suggest_float("ransac_reproj_threshold", 1.5, 4.0, step=0.5)
    cfg["ransac_max_iters"] = trial.suggest_categorical("ransac_max_iters", [2000, 5000, 8000])
    cfg["robust_loss"] = trial.suggest_categorical("robust_loss", ["huber", "soft_l1"])
    cfg["f_scale"] = trial.suggest_float("f_scale", 0.5, 3.0, step=0.5)

    cfg["srt_bounds_profile"] = trial.suggest_categorical("srt_bounds_profile", ["default", "tight"])
    cfg["srt_scale_reg_weight"] = trial.suggest_float("srt_scale_reg_weight", 1e-4, 5e-2, log=True)
    cfg["srt_rot_reg_weight"] = trial.suggest_float("srt_rot_reg_weight", 1e-5, 5e-3, log=True)

    if cfg["adaptive_gauss_fit"]:
        cfg["adaptive_threshold"] = trial.suggest_float("adaptive_threshold", 0.001, 0.01, step=0.001)
        cfg["adaptive_n_sigma"] = trial.suggest_float("adaptive_n_sigma", 2.0, 4.0, step=0.5)
        cfg["adaptive_max_iter"] = trial.suggest_int("adaptive_max_iter", 5, 15, step=5)
    else:
        cfg["fixed_threshold"] = trial.suggest_float("fixed_threshold", 0.003, 0.015, step=0.001)
        cfg["fixed_window_size"] = trial.suggest_categorical("fixed_window_size", [3, 4, 5, 6])

    if cfg["model"] == "full":
        cfg["srt_bounds_profile"] = "default"
        cfg["srt_scale_reg_weight"] = base["srt_scale_reg_weight"]
        cfg["srt_rot_reg_weight"] = base["srt_rot_reg_weight"]

    return cfg


def build_fine_grid(best_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    deltas = {
        "ransac_reproj_threshold": [-0.5, 0.0, 0.5],
        "f_scale": [-0.5, 0.0, 0.5],
        "srt_scale_reg_weight": [0.5, 1.0, 1.5],
        "srt_rot_reg_weight": [0.5, 1.0, 1.5],
    }

    fine = []
    base = dict(best_cfg)
    for dr in deltas["ransac_reproj_threshold"]:
        for df in deltas["f_scale"]:
            cfg = dict(base)
            cfg["ransac_reproj_threshold"] = max(0.5, best_cfg["ransac_reproj_threshold"] + dr)
            cfg["f_scale"] = max(0.1, best_cfg["f_scale"] + df)
            if cfg["model"] == "sRT":
                for ds in deltas["srt_scale_reg_weight"]:
                    for dtheta in deltas["srt_rot_reg_weight"]:
                        c2 = dict(cfg)
                        c2["srt_scale_reg_weight"] = max(1e-5, best_cfg["srt_scale_reg_weight"] * ds)
                        c2["srt_rot_reg_weight"] = max(1e-6, best_cfg["srt_rot_reg_weight"] * dtheta)
                        fine.append(c2)
            else:
                fine.append(cfg)

    out = []
    seen = set()
    for cfg in fine:
        k = json.dumps(cfg, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(cfg)
    return out


def _is_valid_h(H):
    if H is None:
        return False
    H = np.asarray(H)
    if H.shape != (3, 3):
        return False
    return bool(np.all(np.isfinite(H)))


def warp_overlay(im_A, im_B, H_A_to_B):
    img_a = _tensor_to_numpy_image(im_A)
    img_b = _tensor_to_numpy_image(im_B)
    h_b, w_b = img_b.shape[:2]
    if not _is_valid_h(H_A_to_B):
        return img_b.copy()

    warped = cv2.warpPerspective(img_a.astype(np.float32), H_A_to_B.astype(np.float64), (w_b, h_b))
    mask = cv2.warpPerspective(np.ones((img_a.shape[0], img_a.shape[1], 1), dtype=np.float32), H_A_to_B.astype(np.float64), (w_b, h_b))
    if mask.ndim == 2:
        mask = mask[..., None]
    mask = np.clip(mask, 0.0, 1.0)
    alpha = 0.55
    out = img_b * (1.0 - alpha * mask) + warped * (alpha * mask)

    corners = np.array([[0, 0], [img_a.shape[1], 0], [img_a.shape[1], img_a.shape[0]], [0, img_a.shape[0]]], dtype=np.float32).reshape(-1, 1, 2)
    try:
        corners_w = cv2.perspectiveTransform(corners, H_A_to_B.astype(np.float64)).reshape(-1, 2)
        pts = np.round(corners_w).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(1.0, 0.0, 0.0), thickness=2)
    except Exception:
        pass

    return np.clip(out, 0.0, 1.0)


def save_correspondence_overlay(
    out_path: Path,
    im_A,
    im_B,
    pts_A,
    means_B,
    peaks_B,
    in_patch_dim,
    out_patch_dim,
):
    img_a = _tensor_to_numpy_image(im_A)
    img_b = _tensor_to_numpy_image(im_B)
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    sx_a = w_a / float(in_patch_dim)
    sy_a = h_a / float(in_patch_dim)
    sx_b = w_b / float(out_patch_dim)
    sy_b = h_b / float(out_patch_dim)

    pts_a_px = np.column_stack((pts_A[:, 0] * sx_a, pts_A[:, 1] * sy_a)) if len(pts_A) else np.zeros((0, 2))
    means_b_px = np.column_stack((means_B[:, 0] * sx_b, means_B[:, 1] * sy_b)) if len(means_B) else np.zeros((0, 2))
    peaks_b_px = np.column_stack((peaks_B[:, 0] * sx_b, peaks_B[:, 1] * sy_b)) if len(peaks_B) else np.zeros((0, 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img_a)
    ax1.set_title("Image A + pts_A")
    if len(pts_a_px):
        ax1.scatter(pts_a_px[:, 0], pts_a_px[:, 1], s=10, c="cyan")
    ax1.axis("off")

    ax2.imshow(img_b)
    ax2.set_title("Image B + correspondences")
    if len(means_b_px):
        ax2.scatter(means_b_px[:, 0], means_b_px[:, 1], s=10, c="red", label="means")
    if len(peaks_b_px):
        ax2.scatter(peaks_b_px[:, 0], peaks_b_px[:, 1], s=10, c="orange", marker="x", label="peaks")
    ax2.legend(loc="upper right")
    ax2.axis("off")

    if len(pts_a_px) and len(means_b_px):
        n_draw = min(len(pts_a_px), 100)
        idx = np.linspace(0, len(pts_a_px) - 1, n_draw, dtype=int)
        from matplotlib.patches import ConnectionPatch

        for i in idx:
            c = ConnectionPatch(
                xyA=(pts_a_px[i, 0], pts_a_px[i, 1]),
                xyB=(means_b_px[i, 0], means_b_px[i, 1]),
                coordsA="data",
                coordsB="data",
                axesA=ax1,
                axesB=ax2,
                color="white",
                alpha=0.15,
                linewidth=0.8,
            )
            ax2.add_artist(c)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_diff_view(out_path: Path, overlay_init: np.ndarray, overlay_final: np.ndarray):
    diff = np.abs(overlay_final - overlay_init)
    diff_gray = diff.mean(axis=2)
    vmax = max(1e-6, float(np.percentile(diff_gray, 99)))

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    axs[0].imshow(overlay_init)
    axs[0].set_title("RANSAC overlay")
    axs[0].axis("off")

    axs[1].imshow(overlay_final)
    axs[1].set_title("Refined overlay")
    axs[1].axis("off")

    im = axs[2].imshow(diff_gray, cmap="magma", vmin=0.0, vmax=vmax)
    axs[2].set_title("Absolute diff")
    axs[2].axis("off")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_sample(sample: SampleData, cfg: dict[str, Any], write_images_dir: Path | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "sample_id": int(sample.sample_id),
        "status": "ok",
        "error_reason": "",
    }

    t0 = time.perf_counter()
    try:
        pts_A, means_B, peaks_B, covs_B = find_gaussians(
            sample.logits,
            adaptive_gauss_fit=bool(cfg["adaptive_gauss_fit"]),
            plot_heatmaps=False,
            plotter=None,
            log_missing_gaussians=False,
            adaptive_threshold=float(cfg["adaptive_threshold"]),
            adaptive_n_sigma=float(cfg["adaptive_n_sigma"]),
            adaptive_max_iter=int(cfg["adaptive_max_iter"]),
            adaptive_min_half_w=int(cfg["adaptive_min_half_w"]),
            adaptive_max_half_w=int(cfg["adaptive_max_half_w"]),
            fixed_threshold=float(cfg["fixed_threshold"]),
            fixed_window_size=int(cfg["fixed_window_size"]),
        )
    except Exception as exc:
        rec.update(
            {
                "status": "failed",
                "error_reason": f"gaussian_extraction_failed: {exc}",
                "runtime_s": float(time.perf_counter() - t0),
                "num_correspondences": 0,
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "corner_error_ransac_init": float("nan"),
                "corner_error_after_refinement": float("nan"),
                "delta_improvement": float("nan"),
            }
        )
        return rec

    rec["num_correspondences"] = int(len(pts_A))
    rec["in_patch_dim"] = int(sample.logits.shape[-1])
    rec["out_patch_dim"] = int(int(sample.logits.shape[0] ** 0.5))

    if len(pts_A) < 4:
        rec.update(
            {
                "status": "failed",
                "error_reason": "too_few_correspondences_for_homography",
                "runtime_s": float(time.perf_counter() - t0),
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "corner_error_ransac_init": float("nan"),
                "corner_error_after_refinement": float("nan"),
                "delta_improvement": float("nan"),
            }
        )
        return rec

    model = cfg["model"]
    ransac_method = cv2.RANSAC if cfg["ransac_method"] == "RANSAC" else cv2.LMEDS

    srt_bounds = resolve_srt_bounds(str(cfg["srt_bounds_profile"]))

    t_opt0 = time.perf_counter()
    try:
        H_final, H_init, details = optimize_homography(
            pts_A,
            means_B,
            covs_B,
            peaks_B=peaks_B,
            model=model,
            use_means_for_ransac=bool(cfg["use_means_for_ransac"]),
            verbose=0,
            quiet=True,
            ransac_method=ransac_method,
            ransac_reproj_threshold=float(cfg["ransac_reproj_threshold"]),
            ransac_max_iters=int(cfg["ransac_max_iters"]),
            ransac_confidence=float(cfg["ransac_confidence"]),
            robust_loss=str(cfg["robust_loss"]),
            f_scale=float(cfg["f_scale"]),
            max_nfev=int(cfg["max_nfev"]),
            srt_x_scale=cfg["srt_x_scale"],
            srt_bounds=srt_bounds,
            srt_scale_reg_weight=float(cfg["srt_scale_reg_weight"]),
            srt_rot_reg_weight=float(cfg["srt_rot_reg_weight"]),
            return_details=True,
        )
    except Exception as exc:
        rec.update(
            {
                "status": "failed",
                "error_reason": f"optimize_failed: {exc}",
                "runtime_s": float(time.perf_counter() - t0),
                "runtime_opt_s": float(time.perf_counter() - t_opt0),
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "corner_error_ransac_init": float("nan"),
                "corner_error_after_refinement": float("nan"),
                "delta_improvement": float("nan"),
            }
        )
        return rec

    t_opt1 = time.perf_counter()

    H_gt = np.asarray(sample.H_gt, dtype=np.float64)
    img_a_np = _tensor_to_numpy_image(sample.im_A)
    img_b_np = _tensor_to_numpy_image(sample.im_B)
    h_a, w_a = img_a_np.shape[:2]
    h_b, w_b = img_b_np.shape[:2]

    H_final_dl = convert_to_dataloader_homography(
        H_final,
        rec["in_patch_dim"],
        rec["out_patch_dim"],
        crop_res=(h_a, w_a),
        map_res=(h_b, w_b),
    )
    H_init_dl = convert_to_dataloader_homography(
        H_init,
        rec["in_patch_dim"],
        rec["out_patch_dim"],
        crop_res=(h_a, w_a),
        map_res=(h_b, w_b),
    )

    err_init = compute_corner_error(H_gt, H_init_dl, w=w_a, h=h_a)
    err_final = compute_corner_error(H_gt, H_final_dl, w=w_a, h=h_a)

    rec.update(
        {
            "num_inliers": int(details.get("num_inliers", 0)),
            "inlier_ratio": float(details.get("inlier_ratio", 0.0)),
            "optimization_success": bool(details.get("optimization_success", False)),
            "optimization_nfev": int(details.get("optimization_nfev", 0)),
            "corner_error_ransac_init": float(err_init),
            "corner_error_after_refinement": float(err_final),
            "delta_improvement": float(err_init - err_final),
            "runtime_s": float(time.perf_counter() - t0),
            "runtime_opt_s": float(t_opt1 - t_opt0),
            "H_init_condition": float(np.linalg.cond(H_init_dl)) if _is_valid_h(H_init_dl) else float("inf"),
            "H_final_condition": float(np.linalg.cond(H_final_dl)) if _is_valid_h(H_final_dl) else float("inf"),
        }
    )

    if write_images_dir is not None:
        write_images_dir.mkdir(parents=True, exist_ok=True)
        corr_path = write_images_dir / f"sample_{sample.sample_id:06d}_correspondences.png"
        ransac_path = write_images_dir / f"sample_{sample.sample_id:06d}_ransac_overlay.png"
        refined_path = write_images_dir / f"sample_{sample.sample_id:06d}_refined_overlay.png"
        diff_path = write_images_dir / f"sample_{sample.sample_id:06d}_diff_view.png"

        save_correspondence_overlay(
            corr_path,
            sample.im_A,
            sample.im_B,
            pts_A,
            means_B,
            peaks_B,
            in_patch_dim=rec["in_patch_dim"],
            out_patch_dim=rec["out_patch_dim"],
        )

        H_init_px = convert_to_pixel_homography(
            H_init,
            rec["in_patch_dim"],
            rec["out_patch_dim"],
            crop_res=(h_a, w_a),
            map_res=(h_b, w_b),
        )
        H_final_px = convert_to_pixel_homography(
            H_final,
            rec["in_patch_dim"],
            rec["out_patch_dim"],
            crop_res=(h_a, w_a),
            map_res=(h_b, w_b),
        )

        overlay_init = warp_overlay(sample.im_A, sample.im_B, H_init_px)
        overlay_final = warp_overlay(sample.im_A, sample.im_B, H_final_px)
        plt.imsave(ransac_path, overlay_init)
        plt.imsave(refined_path, overlay_final)
        save_diff_view(diff_path, overlay_init, overlay_final)

        rec.update(
            {
                "correspondences_overlay_path": str(corr_path),
                "ransac_overlay_path": str(ransac_path),
                "refined_overlay_path": str(refined_path),
                "diff_view_path": str(diff_path),
            }
        )

    return rec


def apply_triage(records: list[dict[str, Any]], thresholds: TriageThresholds):
    errs = [r.get("corner_error_after_refinement", np.nan) for r in records if r.get("status") == "ok"]
    errs = np.asarray([e for e in errs if np.isfinite(e)], dtype=np.float64)

    outlier_thr = thresholds.max_corner_error_fixed
    if errs.size >= 4:
        q1, q3 = np.percentile(errs, [25, 75])
        iqr = q3 - q1
        iqr_thr = q3 + thresholds.outlier_iqr_k * iqr
        outlier_thr = min(outlier_thr, float(iqr_thr))

    optimize_on = []
    do_not_use = []

    for r in records:
        reasons = []
        if r.get("status") != "ok":
            reasons.append(f"pipeline_failed:{r.get('error_reason', 'unknown')}")
        if int(r.get("num_correspondences", 0)) < thresholds.min_correspondences:
            reasons.append(f"too_few_correspondences<{thresholds.min_correspondences}")
        if int(r.get("num_inliers", 0)) < thresholds.min_inliers:
            reasons.append(f"too_few_inliers<{thresholds.min_inliers}")
        if float(r.get("inlier_ratio", 0.0)) < thresholds.min_inlier_ratio:
            reasons.append(f"low_inlier_ratio<{thresholds.min_inlier_ratio}")

        h_cond = float(r.get("H_final_condition", np.inf))
        if not np.isfinite(h_cond) or h_cond > thresholds.max_h_cond:
            reasons.append(f"degenerate_homography_cond>{thresholds.max_h_cond}")

        ce = float(r.get("corner_error_after_refinement", np.nan))
        if (not np.isfinite(ce)) or ce > outlier_thr:
            reasons.append(f"corner_error_outlier>{outlier_thr:.3f}")

        if not np.isfinite(float(r.get("corner_error_ransac_init", np.nan))):
            reasons.append("invalid_gt_or_corner_error")

        if reasons:
            r["review_status"] = "EXCLUDE"
            r["review_reason"] = "; ".join(reasons)
            do_not_use.append(int(r["sample_id"]))
        else:
            r["review_status"] = "KEEP_FOR_OPTIMIZATION"
            r["review_reason"] = "passed_all_heuristics"
            optimize_on.append(int(r["sample_id"]))

    return sorted(set(optimize_on)), sorted(set(do_not_use))


def summarize_config(config_id: str, cfg: dict[str, Any], sample_records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in sample_records if r.get("status") == "ok"]
    return {
        "config_id": config_id,
        "num_samples": int(len(sample_records)),
        "num_ok": int(len(ok)),
        "corner_error_after_refinement_mean": _safe_mean([r.get("corner_error_after_refinement", np.nan) for r in ok]),
        "corner_error_after_refinement_median": _safe_median([r.get("corner_error_after_refinement", np.nan) for r in ok]),
        "corner_error_ransac_init_mean": _safe_mean([r.get("corner_error_ransac_init", np.nan) for r in ok]),
        "delta_improvement_mean": _safe_mean([r.get("delta_improvement", np.nan) for r in ok]),
        "runtime_s_mean": _safe_mean([r.get("runtime_s", np.nan) for r in ok]),
        "runtime_opt_s_mean": _safe_mean([r.get("runtime_opt_s", np.nan) for r in ok]),
        "num_correspondences_mean": _safe_mean([r.get("num_correspondences", np.nan) for r in ok]),
        "num_inliers_mean": _safe_mean([r.get("num_inliers", np.nan) for r in ok]),
        "inlier_ratio_mean": _safe_mean([r.get("inlier_ratio", np.nan) for r in ok]),
        "params": cfg,
    }


def evaluate_config_on_samples(
    config_id: str,
    cfg: dict[str, Any],
    loaded_samples: list[SampleData],
    run_dir: Path,
    write_review_images: bool,
    per_sample_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
):
    image_dir = run_dir / "review_images" / config_id if write_review_images else None

    sample_records = []
    for sample in loaded_samples:
        rec = run_sample(sample, cfg, write_images_dir=image_dir)
        rec["config_id"] = config_id
        rec["config"] = cfg
        sample_records.append(rec)
        per_sample_rows.append(rec)

    agg = summarize_config(config_id=config_id, cfg=cfg, sample_records=sample_records)
    aggregate_rows.append(agg)
    return agg, sample_records


def rank_aggregates(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(x):
        ce = x.get("corner_error_after_refinement_mean", float("inf"))
        rt = x.get("runtime_s_mean", float("inf"))
        return (
            ce if np.isfinite(ce) else float("inf"),
            rt if np.isfinite(rt) else float("inf"),
            -x.get("num_ok", 0),
        )

    ranked = sorted(aggregates, key=key)
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    return ranked


def save_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def save_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            flat = {}
            for k in keys:
                v = r.get(k)
                if isinstance(v, (dict, list, tuple)):
                    flat[k] = json.dumps(v, sort_keys=True)
                else:
                    flat[k] = v
            w.writerow(flat)


def write_summary_report(
    out_path: Path,
    baseline_agg: dict[str, Any] | None,
    best_agg: dict[str, Any] | None,
    ranked: list[dict[str, Any]],
    per_sample_rows: list[dict[str, Any]],
):
    lines = []
    lines.append("# SAT-ROMA Ablation Report")
    lines.append("")

    if baseline_agg:
        lines.append("## Baseline")
        lines.append(f"- config_id: `{baseline_agg['config_id']}`")
        lines.append(f"- corner_error_after_refinement_mean: {baseline_agg['corner_error_after_refinement_mean']:.6f}")
        lines.append(f"- corner_error_ransac_init_mean: {baseline_agg['corner_error_ransac_init_mean']:.6f}")
        lines.append(f"- delta_improvement_mean: {baseline_agg['delta_improvement_mean']:.6f}")
        lines.append(f"- runtime_s_mean: {baseline_agg['runtime_s_mean']:.6f}")
        lines.append("")

    if best_agg:
        lines.append("## Best Config")
        lines.append(f"- config_id: `{best_agg['config_id']}`")
        lines.append(f"- rank: {best_agg.get('rank', 'n/a')}")
        lines.append(f"- corner_error_after_refinement_mean: {best_agg['corner_error_after_refinement_mean']:.6f}")
        lines.append(f"- runtime_s_mean: {best_agg['runtime_s_mean']:.6f}")
        lines.append(f"- params: `{json.dumps(best_agg.get('params', {}), sort_keys=True)}`")
        lines.append("")

    if baseline_agg and best_agg:
        b = baseline_agg["corner_error_after_refinement_mean"]
        c = best_agg["corner_error_after_refinement_mean"]
        lines.append("## Baseline vs Best")
        lines.append(f"- corner_error_delta (baseline - best): {b - c:.6f}")
        lines.append("")

    if ranked:
        lines.append("## Top Configs")
        for r in ranked[:5]:
            lines.append(
                f"- rank {r['rank']} | {r['config_id']} | ce_mean={r['corner_error_after_refinement_mean']:.6f} | runtime_mean={r['runtime_s_mean']:.6f}"
            )
        lines.append("")

    if per_sample_rows:
        lines.append("## Largest Improvements")
        by_imp = [r for r in per_sample_rows if np.isfinite(_to_float(r.get("delta_improvement")))]
        by_imp.sort(key=lambda x: _to_float(x.get("delta_improvement"), default=-1e18), reverse=True)
        for r in by_imp[:5]:
            lines.append(
                f"- sample {r['sample_id']} ({r['config_id']}): delta_improvement={_to_float(r['delta_improvement']):.6f}, final={_to_float(r['corner_error_after_refinement']):.6f}"
            )
        lines.append("")

        lines.append("## Largest Regressions")
        by_imp.sort(key=lambda x: _to_float(x.get("delta_improvement"), default=1e18))
        for r in by_imp[:5]:
            lines.append(
                f"- sample {r['sample_id']} ({r['config_id']}): delta_improvement={_to_float(r['delta_improvement']):.6f}, final={_to_float(r['corner_error_after_refinement']):.6f}"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="SAT-ROMA homography ablation runner")
    parser.add_argument("--dataset-dir", required=True, help="Folder with input_sample_*.pt and sample_*_tensor.pt")
    parser.add_argument("--sample-ids", default="", help="Comma list, range, or single id. Empty means auto-discover")
    parser.add_argument("--split-file", default="", help="Optional file with sample ids (one per line or comma-separated)")
    parser.add_argument("--mode", choices=["baseline", "grid", "both", "optuna"], default="both")
    parser.add_argument("--max-grid-configs", type=int, default=64)
    parser.add_argument("--run-fine-search", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=40)
    parser.add_argument("--optuna-startup-trials", type=int, default=10)
    parser.add_argument("--optuna-n-jobs", type=int, default=1)
    parser.add_argument("--optuna-storage", default="", help="Optional Optuna storage URL, e.g. sqlite:///optuna.db")
    parser.add_argument("--optuna-study-name", default="sat_roma_homography")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-root", default="experiments/sat_roma_runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--write-review-images", action="store_true")
    args = parser.parse_args()

    set_deterministic_seeds(args.seed)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    if args.sample_ids.strip():
        sample_ids = parse_sample_ids(args.sample_ids)
    elif args.split_file.strip():
        sample_ids = load_ids_from_split_file(Path(args.split_file))
    else:
        sample_ids = discover_sample_ids(dataset_dir)

    if not sample_ids:
        raise RuntimeError("No sample ids found. Pass --sample-ids or provide matching tensor files.")

    run_name = args.run_name.strip() or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    configs: list[dict[str, Any]] = []
    if args.mode in {"baseline", "both"}:
        configs.append(default_config())
    if args.mode in {"grid", "both"}:
        for cfg in create_coarse_grid(seed=args.seed, max_configs=args.max_grid_configs):
            if not any(json.dumps(c, sort_keys=True) == json.dumps(cfg, sort_keys=True) for c in configs):
                configs.append(cfg)

    loaded_samples = []
    for sid in sample_ids:
        loaded_samples.append(load_sample(dataset_dir, sid))

    per_sample_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for i, cfg in enumerate(configs):
        config_id = f"cfg_{i:03d}"
        evaluate_config_on_samples(
            config_id=config_id,
            cfg=cfg,
            loaded_samples=loaded_samples,
            run_dir=run_dir,
            write_review_images=args.write_review_images,
            per_sample_rows=per_sample_rows,
            aggregate_rows=aggregate_rows,
        )

    optuna_trials_table = []
    if args.mode == "optuna":
        try:
            import optuna
        except Exception as exc:
            raise RuntimeError("Optuna mode requested but optuna is not installed.") from exc

        base = default_config()
        sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.optuna_startup_trials)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=max(3, args.optuna_startup_trials // 2))
        storage = args.optuna_storage.strip() or None
        study = optuna.create_study(
            study_name=args.optuna_study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=bool(storage),
        )

        trial_counter = {"value": 0}

        def objective(trial):
            trial_idx = trial_counter["value"]
            trial_counter["value"] += 1
            cfg = suggest_optuna_config(trial, base=base)
            config_id = f"optuna_trial_{trial_idx:03d}"

            agg, sample_records = evaluate_config_on_samples(
                config_id=config_id,
                cfg=cfg,
                loaded_samples=loaded_samples,
                run_dir=run_dir,
                write_review_images=args.write_review_images,
                per_sample_rows=per_sample_rows,
                aggregate_rows=aggregate_rows,
            )
            num_fail = int(agg["num_samples"] - agg["num_ok"])
            ce_mean = float(agg["corner_error_after_refinement_mean"])
            objective_value = ce_mean + (num_fail * 10000.0)
            trial.set_user_attr("config_id", config_id)
            trial.set_user_attr("num_ok", int(agg["num_ok"]))
            trial.set_user_attr("num_samples", int(agg["num_samples"]))
            trial.set_user_attr("ce_mean", ce_mean)
            trial.set_user_attr("runtime_mean", float(agg["runtime_s_mean"]))
            trial.report(objective_value, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            _ = sample_records
            return objective_value

        study.optimize(objective, n_trials=args.optuna_trials, n_jobs=args.optuna_n_jobs, show_progress_bar=False)

        for t in study.trials:
            optuna_trials_table.append(
                {
                    "trial_number": int(t.number),
                    "state": str(t.state),
                    "value": float(t.value) if t.value is not None else float("nan"),
                    "params": t.params,
                    "config_id": t.user_attrs.get("config_id", ""),
                    "num_ok": t.user_attrs.get("num_ok", 0),
                    "num_samples": t.user_attrs.get("num_samples", 0),
                    "ce_mean": t.user_attrs.get("ce_mean", float("nan")),
                    "runtime_mean": t.user_attrs.get("runtime_mean", float("nan")),
                }
            )

    ranked = rank_aggregates(aggregate_rows)
    best_agg = ranked[0] if ranked else None

    baseline_agg = None
    if configs:
        baseline_id = "cfg_000"
        for row in ranked:
            if row["config_id"] == baseline_id:
                baseline_agg = row
                break

    thresholds = TriageThresholds()

    if args.run_fine_search and best_agg is not None:
        fine_cfgs = build_fine_grid(best_agg["params"])
        start_idx = len(configs)
        for j, cfg in enumerate(fine_cfgs):
            config_id = f"cfg_{start_idx + j:03d}_fine"
            evaluate_config_on_samples(
                config_id=config_id,
                cfg=cfg,
                loaded_samples=loaded_samples,
                run_dir=run_dir,
                write_review_images=args.write_review_images,
                per_sample_rows=per_sample_rows,
                aggregate_rows=aggregate_rows,
            )

        ranked = rank_aggregates(aggregate_rows)
        best_agg = ranked[0] if ranked else best_agg

    best_id = best_agg["config_id"] if best_agg else None
    best_records = [r for r in per_sample_rows if r.get("config_id") == best_id] if best_id else []
    optimize_on, do_not_use = apply_triage(best_records, thresholds)

    review_rows = []
    for r in best_records:
        review_rows.append(
            {
                "sample_id": r["sample_id"],
                "config_id": r["config_id"],
                "corner_error_ransac_init": r.get("corner_error_ransac_init"),
                "corner_error_after_refinement": r.get("corner_error_after_refinement"),
                "delta_improvement": r.get("delta_improvement"),
                "runtime_s": r.get("runtime_s"),
                "num_correspondences": r.get("num_correspondences"),
                "num_inliers": r.get("num_inliers"),
                "inlier_ratio": r.get("inlier_ratio"),
                "correspondences_overlay_path": r.get("correspondences_overlay_path", ""),
                "ransac_overlay_path": r.get("ransac_overlay_path", ""),
                "refined_overlay_path": r.get("refined_overlay_path", ""),
                "diff_view_path": r.get("diff_view_path", ""),
                "status_label": r.get("review_status", "EXCLUDE"),
                "reason": r.get("review_reason", ""),
            }
        )

    output_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dataset_dir": str(dataset_dir),
        "sample_ids": sample_ids,
        "mode": args.mode,
        "max_grid_configs": args.max_grid_configs,
        "run_fine_search": bool(args.run_fine_search),
        "optuna_trials": args.optuna_trials,
        "optuna_startup_trials": args.optuna_startup_trials,
        "optuna_n_jobs": args.optuna_n_jobs,
        "optuna_storage": args.optuna_storage,
        "optuna_study_name": args.optuna_study_name,
        "thresholds": thresholds.__dict__,
        "best_config_id": best_agg["config_id"] if best_agg else None,
        "optimize_on": optimize_on,
        "do_not_use_for_tuning": do_not_use,
    }

    save_json(run_dir / "run_meta.json", output_payload)
    save_json(run_dir / "configs_ranked.json", ranked)
    save_json(run_dir / "per_sample_metrics.json", per_sample_rows)
    save_json(run_dir / "review_index.json", review_rows)
    if optuna_trials_table:
        save_json(run_dir / "optuna_trials.json", optuna_trials_table)
        save_csv(run_dir / "optuna_trials.csv", optuna_trials_table)
    save_csv(run_dir / "configs_ranked.csv", ranked)
    save_csv(run_dir / "per_sample_metrics.csv", per_sample_rows)
    save_csv(run_dir / "review_index.csv", review_rows)

    (run_dir / "optimize_on.txt").write_text("\n".join(map(str, optimize_on)) + ("\n" if optimize_on else ""), encoding="utf-8")
    (run_dir / "do_not_use_for_tuning.txt").write_text(
        "\n".join(map(str, do_not_use)) + ("\n" if do_not_use else ""), encoding="utf-8"
    )

    write_summary_report(
        out_path=run_dir / "summary_report.md",
        baseline_agg=baseline_agg,
        best_agg=best_agg,
        ranked=ranked,
        per_sample_rows=per_sample_rows,
    )

    print(f"Run directory: {run_dir}")
    if best_agg is not None:
        print(
            "Best config:",
            best_agg["config_id"],
            "corner_error_after_refinement_mean=",
            f"{best_agg['corner_error_after_refinement_mean']:.6f}",
            "runtime_s_mean=",
            f"{best_agg['runtime_s_mean']:.6f}",
        )
    print(f"optimize_on count: {len(optimize_on)}")
    print(f"do_not_use_for_tuning count: {len(do_not_use)}")


if __name__ == "__main__":
    main()
