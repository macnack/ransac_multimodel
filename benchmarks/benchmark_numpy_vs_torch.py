import argparse
import os
import platform
import time
from dataclasses import dataclass

import numpy as np

from ransac_multimodel.parity_utils import (
    now_iso_utc,
    np_to_torch,
    percentile_ms,
    resolve_device,
    set_deterministic_seeds,
    torch_to_np,
    write_json,
)


@dataclass
class SampleCase:
    sample_id: int
    pts_A: np.ndarray
    means_B: np.ndarray
    peaks_B: np.ndarray
    covs_B: np.ndarray
    H_gt: np.ndarray | None


def parse_sample_ids(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return [98, 122, 128]
    if "-" in raw:
        lo_s, hi_s = raw.split("-", maxsplit=1)
        lo, hi = int(lo_s), int(hi_s)
        if lo > hi:
            raise ValueError("Invalid sample range: start > end")
        return list(range(lo, hi + 1))
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("No sample ids parsed")
    return out


def load_case_from_tensors(sample_id: int, tensors_dir: str, quiet: bool = False) -> SampleCase:
    import torch
    from ransac_multimodel.correspondence import find_gaussians

    in_path = os.path.join(tensors_dir, f"input_sample_{sample_id:06d}.pt")
    sample_path = os.path.join(tensors_dir, f"sample_{sample_id:03d}_tensor.pt")

    gt = torch.load(in_path, map_location=torch.device("cpu"))
    sample = torch.load(sample_path, map_location=torch.device("cpu"))

    logits = sample[16]["gm_cls"][0]
    pts_A, means_B, peaks_B, covs_B = find_gaussians(
        logits,
        adaptive_gauss_fit=True,
        plot_heatmaps=False,
        plotter=None,
        log_missing_gaussians=not quiet,
    )

    H_gt = gt["homography_gt"]
    H_gt = H_gt.numpy() if hasattr(H_gt, "numpy") else np.asarray(H_gt)

    return SampleCase(
        sample_id=sample_id,
        pts_A=pts_A,
        means_B=means_B,
        peaks_B=peaks_B,
        covs_B=covs_B,
        H_gt=H_gt,
    )


def build_synthetic_case(seed: int = 1234, n_points: int = 196) -> SampleCase:
    if n_points <= 0:
        raise ValueError("n_points must be > 0")

    rng = np.random.default_rng(seed)
    size_a = 14

    # sRT ground truth
    s = 1.08
    theta = np.radians(8.0)
    tx = 2.2
    ty = -1.4
    c, si = np.cos(theta), np.sin(theta)
    H_gt = np.array(
        [
            [s * c, -s * si, tx],
            [s * si, s * c, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    pts_A = rng.uniform(low=0.0, high=float(size_a), size=(n_points, 2)).astype(np.float64)

    ones = np.ones((pts_A.shape[0], 1), dtype=np.float64)
    proj = (H_gt @ np.hstack([pts_A, ones]).T).T
    means_B = proj[:, :2] / (proj[:, 2:] + 1e-6)
    means_B = means_B + rng.normal(0.0, 0.03, size=means_B.shape)

    peaks_B = means_B.copy()
    covs_B = np.repeat(np.eye(2, dtype=np.float64)[None, :, :] * 0.4, n_points, axis=0)

    return SampleCase(
        sample_id=-1,
        pts_A=pts_A.astype(np.float32),
        means_B=means_B.astype(np.float32),
        peaks_B=peaks_B.astype(np.float32),
        covs_B=covs_B.astype(np.float32),
        H_gt=H_gt.astype(np.float32),
    )


def prepare_inv_covs(covs_B: np.ndarray) -> np.ndarray:
    covs_safe = covs_B + np.eye(2) * 1e-6
    return np.linalg.inv(covs_safe)


def _sync_if_cuda(device: str):
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def timeit_ms(fn, repeats: int, warmup: int, device: str = "cpu"):
    for _ in range(max(warmup, 0)):
        fn()
    times = []
    for _ in range(max(repeats, 1)):
        _sync_if_cuda(device)
        t0 = time.perf_counter()
        fn()
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return {
        "runs": len(times),
        "median_ms": float(np.median(times)),
        "p95_ms": percentile_ms(times, 95),
        "all_ms": times,
    }


def benchmark_residual(case: SampleCase, repeats: int, warmup: int, device: str):
    try:
        from ransac_multimodel.homography import homography_residuals_vectorized
    except Exception as exc:
        torch_results = {"cpu": {"status": "skipped", "reason": str(exc)}}
        if device == "cuda":
            torch_results["cuda"] = {"status": "skipped", "reason": str(exc)}
        return {
            "numpy_cpu": {"status": "skipped", "reason": str(exc)},
            "torch": torch_results,
        }

    inv_covs = prepare_inv_covs(case.covs_B)

    H_init_np = np.eye(3, dtype=np.float64)
    h_elements_np = H_init_np.flatten()[:8]

    def run_numpy():
        return homography_residuals_vectorized(h_elements_np, case.pts_A, case.means_B, inv_covs)

    numpy_timing = timeit_ms(run_numpy, repeats=repeats, warmup=warmup, device="cpu")
    np_out = run_numpy()

    try:
        from ransac_multimodel.homography_torch import homography_residuals_vectorized_torch
    except Exception as exc:
        torch_results = {"cpu": {"status": "skipped", "reason": str(exc)}}
        if device == "cuda":
            torch_results["cuda"] = {"status": "skipped", "reason": str(exc)}
        return {"numpy_cpu": numpy_timing, "torch": torch_results}

    torch_results = {}
    for dev in ["cpu"] + (["cuda"] if device == "cuda" else []):
        try:
            h_t = np_to_torch(h_elements_np, device=dev)
            pts_t = np_to_torch(case.pts_A, device=dev)
            means_t = np_to_torch(case.means_B, device=dev)
            inv_covs_t = np_to_torch(inv_covs, device=dev)

            def run_torch():
                return homography_residuals_vectorized_torch(h_t, pts_t, means_t, inv_covs_t)

            timing = timeit_ms(run_torch, repeats=repeats, warmup=warmup, device=dev)
            torch_out = torch_to_np(run_torch())
            diff = np.abs(np_out - torch_out)

            torch_results[dev] = {
                "timing": timing,
                "parity": {
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                    "allclose": bool(np.allclose(np_out, torch_out, rtol=1e-6, atol=1e-7)),
                },
            }
        except Exception as exc:
            torch_results[dev] = {"status": "skipped", "reason": str(exc)}

    return {"numpy_cpu": numpy_timing, "torch": torch_results}


def benchmark_end2end(case: SampleCase, repeats: int, warmup: int, model: str, device: str, quiet: bool):
    try:
        from ransac_multimodel.homography import compute_corner_error, optimize_homography
    except Exception as exc:
        torch_results = {"cpu": {"status": "skipped", "reason": str(exc)}}
        if device == "cuda":
            torch_results["cuda"] = {"status": "skipped", "reason": str(exc)}
        return {
            "numpy_cpu": {"status": "skipped", "reason": str(exc)},
            "torch": torch_results,
        }

    def run_numpy():
        return optimize_homography(
            case.pts_A,
            case.means_B,
            case.covs_B,
            peaks_B=case.peaks_B,
            model=model,
            verbose=0,
            quiet=quiet,
        )

    numpy_timing = timeit_ms(run_numpy, repeats=repeats, warmup=warmup, device="cpu")
    H_np, _ = run_numpy()

    try:
        from ransac_multimodel.homography_torch import optimize_homography_torch
    except Exception as exc:
        torch_results = {"cpu": {"status": "skipped", "reason": str(exc)}}
        if device == "cuda":
            torch_results["cuda"] = {"status": "skipped", "reason": str(exc)}
        return {"numpy_cpu": numpy_timing, "torch": torch_results}

    torch_results = {}
    for dev in ["cpu"] + (["cuda"] if device == "cuda" else []):
        try:
            def run_torch():
                return optimize_homography_torch(
                    case.pts_A,
                    case.means_B,
                    case.covs_B,
                    peaks_B=case.peaks_B,
                    model=model,
                    verbose=0,
                    device=dev,
                    quiet=quiet,
                )

            timing = timeit_ms(run_torch, repeats=repeats, warmup=warmup, device=dev)
            H_t, _ = run_torch()

            H_np_norm = H_np / H_np[2, 2]
            H_t_norm = H_t / H_t[2, 2]
            h_diff = np.linalg.norm(H_np_norm - H_t_norm)

            parity = {
                "H_fro_diff": float(h_diff),
            }
            if case.H_gt is not None:
                ce_np = compute_corner_error(case.H_gt, H_np_norm, w=1024, h=1024)
                ce_t = compute_corner_error(case.H_gt, H_t_norm, w=1024, h=1024)
                parity.update(
                    {
                        "corner_error_numpy": float(ce_np),
                        "corner_error_torch": float(ce_t),
                        "corner_error_delta": float(abs(ce_np - ce_t)),
                    }
                )

            torch_results[dev] = {
                "timing": timing,
                "parity": parity,
            }
        except Exception as exc:
            torch_results[dev] = {"status": "skipped", "reason": str(exc)}

    return {"numpy_cpu": numpy_timing, "torch": torch_results}


def summarize_to_console(results):
    print("\n=== Benchmark Summary ===")
    for case in results["cases"]:
        sid = case["sample_id"]
        print(f"\nSample {sid}:")
        if "residual" in case:
            np_res = case["residual"]["numpy_cpu"]
            if "median_ms" in np_res:
                n = np_res["median_ms"]
                print(f"  residual numpy cpu median: {n:.3f} ms")
                for dev, rec in case["residual"]["torch"].items():
                    if "timing" in rec:
                        t = rec["timing"]["median_ms"]
                        speed = n / t if t > 0 else float("inf")
                        print(f"  residual torch {dev} median: {t:.3f} ms (speedup x{speed:.2f})")
                    else:
                        print(f"  residual torch {dev}: skipped ({rec.get('reason', 'unknown')})")
            else:
                print(f"  residual numpy cpu: skipped ({np_res.get('reason', 'unknown')})")
                for dev, rec in case["residual"]["torch"].items():
                    print(f"  residual torch {dev}: skipped ({rec.get('reason', 'unknown')})")

        if "end2end" in case:
            np_e2e = case["end2end"]["numpy_cpu"]
            if "median_ms" in np_e2e:
                n = np_e2e["median_ms"]
                print(f"  end2end numpy cpu median: {n:.3f} ms")
                for dev, rec in case["end2end"]["torch"].items():
                    if "timing" in rec:
                        t = rec["timing"]["median_ms"]
                        speed = n / t if t > 0 else float("inf")
                        delta = rec["parity"].get("corner_error_delta", float("nan"))
                        ce_np = rec["parity"].get("corner_error_numpy", float("nan"))
                        ce_t = rec["parity"].get("corner_error_torch", float("nan"))
                        print(
                            f"  end2end torch {dev} median: {t:.3f} ms "
                            f"(speedup x{speed:.2f}, "
                            f"corner_error_numpy={ce_np:.6f}, "
                            f"corner_error_torch={ce_t:.6f}, "
                            f"corner_error_delta={delta:.6f})"
                        )
                    else:
                        print(f"  end2end torch {dev}: skipped ({rec.get('reason', 'unknown')})")
            else:
                print(f"  end2end numpy cpu: skipped ({np_e2e.get('reason', 'unknown')})")
                for dev, rec in case["end2end"]["torch"].items():
                    print(f"  end2end torch {dev}: skipped ({rec.get('reason', 'unknown')})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark NumPy vs Torch homography paths.")
    parser.add_argument("--sample-ids", default="98,122,128", help="Comma list (98,122) or range (98-128)")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--modes", choices=["residual", "end2end", "all"], default="all")
    parser.add_argument("--model", choices=["full", "sRT"], default="sRT")
    parser.add_argument("--tensors-dir", default="./tensors")
    parser.add_argument("--synthetic", action="store_true", help="Run benchmark on synthetic correspondences")
    parser.add_argument("--synthetic-n", type=int, default=196, help="Number of synthetic correspondences")
    parser.add_argument("--output", default="benchmarks/results/latest.json")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--quiet", action="store_true", help="Suppress noisy per-iteration logs")
    args = parser.parse_args()

    set_deterministic_seeds(args.seed)

    resolved_device = resolve_device(args.device)
    if args.device == "cuda" and resolved_device != "cuda":
        raise RuntimeError("CUDA explicitly requested but unavailable")

    if args.device == "auto" and resolved_device != "cuda":
        print("CUDA not available: GPU benchmarks will be skipped.")

    sample_ids = parse_sample_ids(args.sample_ids)

    cases = []
    if args.synthetic:
        cases = [build_synthetic_case(seed=args.seed, n_points=args.synthetic_n)]
    else:
        for sid in sample_ids:
            cases.append(load_case_from_tensors(sid, args.tensors_dir, quiet=args.quiet))

    out = {
        "timestamp_utc": now_iso_utc(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "resolved_device": resolved_device,
        },
        "config": {
            "sample_ids": sample_ids,
            "modes": args.modes,
            "model": args.model,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "synthetic": args.synthetic,
            "synthetic_n": args.synthetic_n,
            "quiet": args.quiet,
        },
        "cases": [],
    }

    for case in cases:
        case_rec = {"sample_id": int(case.sample_id), "num_points": int(case.pts_A.shape[0])}
        if args.modes in {"residual", "all"}:
            case_rec["residual"] = benchmark_residual(
                case,
                repeats=args.repeats,
                warmup=args.warmup,
                device=resolved_device,
            )
        if args.modes in {"end2end", "all"}:
            case_rec["end2end"] = benchmark_end2end(
                case,
                repeats=args.repeats,
                warmup=args.warmup,
                model=args.model,
                device=resolved_device,
                quiet=args.quiet,
            )
        out["cases"].append(case_rec)

    summarize_to_console(out)

    write_json(args.output, out)
    ts_name = os.path.join(
        os.path.dirname(args.output),
        f"run_{int(time.time())}.json",
    )
    write_json(ts_name, out)
    print(f"\nWrote benchmark results to: {args.output}")
    print(f"Wrote benchmark results snapshot to: {ts_name}")


if __name__ == "__main__":
    main()
