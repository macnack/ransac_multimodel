import unittest

import numpy as np


def _has_module(name: str) -> bool:
    return __import__("importlib").util.find_spec(name) is not None


class TestTorchParity(unittest.TestCase):
    @unittest.skipUnless(_has_module("torch") and _has_module("cv2"), "torch and cv2 are required")
    def test_project_points_parity(self):
        import torch

        from ransac_multimodel.homography import project_points
        from ransac_multimodel.homography_torch import project_points_torch

        rng = np.random.default_rng(1234)
        H = np.array(
            [[1.03, -0.04, 2.1], [0.02, 0.99, -1.3], [0.0004, -0.0002, 1.0]],
            dtype=np.float64,
        )
        pts = rng.uniform(low=0.0, high=64.0, size=(256, 2)).astype(np.float64)

        np_out = project_points(H, pts)

        H_t = torch.as_tensor(H, dtype=torch.float64)
        pts_t = torch.as_tensor(pts, dtype=torch.float64)
        t_out = project_points_torch(H_t, pts_t).detach().cpu().numpy()

        self.assertTrue(np.allclose(np_out, t_out, rtol=1e-7, atol=1e-8))

    @unittest.skipUnless(_has_module("torch") and _has_module("cv2"), "torch and cv2 are required")
    def test_residual_parity(self):
        import torch

        from ransac_multimodel.homography import homography_residuals_vectorized
        from ransac_multimodel.homography_torch import homography_residuals_vectorized_torch

        rng = np.random.default_rng(4321)
        pts_A = rng.uniform(low=0.0, high=14.0, size=(196, 2)).astype(np.float64)
        means_B = rng.uniform(low=0.0, high=64.0, size=(196, 2)).astype(np.float64)

        covs = np.repeat(np.eye(2, dtype=np.float64)[None, :, :] * 0.6, 196, axis=0)
        covs += rng.normal(0.0, 0.01, size=covs.shape)
        for i in range(covs.shape[0]):
            covs[i] = covs[i] @ covs[i].T + np.eye(2) * 1e-3

        inv_covs = np.linalg.inv(covs + np.eye(2) * 1e-6)

        h_elements = np.array([1.0, 0.0, 0.5, 0.0, 1.0, -0.3, 0.0, 0.0], dtype=np.float64)

        np_out = homography_residuals_vectorized(h_elements, pts_A, means_B, inv_covs)

        t_out = homography_residuals_vectorized_torch(
            torch.as_tensor(h_elements, dtype=torch.float64),
            torch.as_tensor(pts_A, dtype=torch.float64),
            torch.as_tensor(means_B, dtype=torch.float64),
            torch.as_tensor(inv_covs, dtype=torch.float64),
        ).detach().cpu().numpy()

        self.assertTrue(np.allclose(np_out, t_out, rtol=1e-6, atol=1e-7))

    @unittest.skipUnless(_has_module("torch") and _has_module("cv2"), "torch and cv2 are required")
    def test_end2end_corner_error_delta(self):
        from benchmarks.benchmark_numpy_vs_torch import build_synthetic_case
        from ransac_multimodel.homography import compute_corner_error, optimize_homography
        from ransac_multimodel.homography_torch import optimize_homography_torch

        case = build_synthetic_case(seed=1234)

        H_np, _ = optimize_homography(
            case.pts_A,
            case.means_B,
            case.covs_B,
            peaks_B=case.peaks_B,
            model="sRT",
            verbose=0,
        )

        H_t, _ = optimize_homography_torch(
            case.pts_A,
            case.means_B,
            case.covs_B,
            peaks_B=case.peaks_B,
            model="sRT",
            device="cpu",
            verbose=0,
            max_iter=150,
        )

        H_np = H_np / H_np[2, 2]
        H_t = H_t / H_t[2, 2]

        ce_np = compute_corner_error(case.H_gt, H_np, w=1024, h=1024)
        ce_t = compute_corner_error(case.H_gt, H_t, w=1024, h=1024)

        self.assertLess(abs(ce_np - ce_t), 1.0)


if __name__ == "__main__":
    unittest.main()
