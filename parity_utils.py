import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np


def set_deterministic_seeds(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def resolve_device(device: str = "auto") -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def np_to_torch(array, device: str = "cpu", dtype=None):
    import torch

    if dtype is None:
        dtype = torch.float64
    return torch.as_tensor(array, dtype=dtype, device=device)


def torch_to_np(tensor):
    return tensor.detach().cpu().numpy()


def percentile_ms(values, q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_json(path: str, payload) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def resolve_srt_bounds(profile: str) -> tuple[list[float], list[float]]:
    if profile == "tight":
        return ([0.50, np.radians(-90), -30, -30], [2.0, np.radians(90), 30, 30])
    return ([0.25, np.radians(-180), -60, -60], [3.0, np.radians(180), 60, 60])


def gaussian_config(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(cfg or {})
    return {
        "adaptive_gauss_fit": bool(cfg.get("adaptive_gauss_fit", True)),
        "adaptive_threshold": float(cfg.get("adaptive_threshold", 0.003)),
        "adaptive_n_sigma": float(cfg.get("adaptive_n_sigma", 3.0)),
        "adaptive_max_iter": int(cfg.get("adaptive_max_iter", 10)),
        "adaptive_min_half_w": int(cfg.get("adaptive_min_half_w", 1)),
        "adaptive_max_half_w": int(cfg.get("adaptive_max_half_w", 5)),
        "fixed_threshold": float(cfg.get("fixed_threshold", 0.008)),
        "fixed_window_size": int(cfg.get("fixed_window_size", 4)),
    }


def optimize_params(cfg: Mapping[str, Any] | None = None, *, quiet: bool = True) -> dict[str, Any]:
    import cv2

    cfg = dict(cfg or {})
    ransac_method_name = str(cfg.get("ransac_method", "RANSAC")).upper()
    ransac_method = cv2.RANSAC if ransac_method_name == "RANSAC" else cv2.LMEDS
    srt_bounds_profile = str(cfg.get("srt_bounds_profile", "default"))
    return {
        "model": str(cfg.get("model", "sRT")),
        "use_means_for_ransac": bool(cfg.get("use_means_for_ransac", False)),
        "quiet": bool(quiet),
        "ransac_method": ransac_method,
        "ransac_reproj_threshold": float(cfg.get("ransac_reproj_threshold", 3.0)),
        "ransac_max_iters": int(cfg.get("ransac_max_iters", 5000)),
        "ransac_confidence": float(cfg.get("ransac_confidence", 0.995)),
        "robust_loss": str(cfg.get("robust_loss", "huber")),
        "f_scale": float(cfg.get("f_scale", 2.0)),
        "max_nfev": int(cfg.get("max_nfev", 5000)),
        "srt_x_scale": cfg.get("srt_x_scale", [0.20, 1.0, 1.0, 1.0]),
        "srt_bounds": resolve_srt_bounds(srt_bounds_profile),
        "srt_scale_reg_weight": float(cfg.get("srt_scale_reg_weight", 0.01)),
        "srt_rot_reg_weight": float(cfg.get("srt_rot_reg_weight", 0.001)),
    }
