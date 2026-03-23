import cv2
import numpy as np


def extract_gaussians_adaptive(
    heatmap,
    threshold=0.1,
    init_window=5,
    n_sigma=3.0,
    max_iter=2,
    min_half_w=2,
    max_half_w=30,
):
    """
    Like extract_gaussians_from_heatmap2 but with an adaptive window size.

    For each detected peak the window is grown iteratively to cover n_sigma
    standard deviations of the fitted Gaussian. This handles both sharp,
    narrow blobs and wide, diffuse blobs without choosing a fixed window_size.
    """
    if heatmap.dtype != np.float32:
        heatmap = heatmap.astype(np.float32)

    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)

    # global peak for ransac initialization (falls back to local peaks if key is absent for old data)
    global_peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    global_peak = np.array([float(global_peak_idx[1]), float(global_peak_idx[0])])  # (x, y) format

    # peak detection with the initial window size
    kernel = np.ones((init_window, init_window), np.uint8)
    local_max = cv2.dilate(heatmap, kernel)
    peaks = (heatmap > threshold) & (local_max == heatmap)
    y_peaks, x_peaks = np.where(peaks)

    h, w = heatmap.shape
    gaussians = []

    def _fit_window(px, py, half_w):
        """Fit a Gaussian in the window around (px, py) with given half_w."""
        y_min = max(0, py - half_w)
        y_max = min(h, py + half_w + 1)
        x_min = max(0, px - half_w)
        x_max = min(w, px + half_w + 1)
        window = heatmap[y_min:y_max, x_min:x_max]

        M = cv2.moments(window)
        sum_p = M["m00"]
        if sum_p == 0:
            return None

        mean_x = x_min + M["m10"] / sum_p
        mean_y = y_min + M["m01"] / sum_p

        cov_xx = M["mu20"] / sum_p
        cov_yy = M["mu02"] / sum_p
        cov_xy = M["mu11"] / sum_p

        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
        cov += np.eye(2) * 1e-4
        return mean_x, mean_y, cov, sum_p

    for px, py in zip(x_peaks, y_peaks):
        half_w = init_window // 2
        result = _fit_window(px, py, half_w)
        if result is None:
            continue

        # iteratively adapt the window to the fitted sigma
        for _ in range(max_iter):
            mean_x, mean_y, cov, sum_p = result

            eigvals = np.linalg.eigvalsh(cov)
            sigma = np.sqrt(max(eigvals))

            new_half_w = int(np.ceil(n_sigma * sigma))
            new_half_w = int(np.clip(new_half_w, min_half_w, max_half_w))

            if new_half_w == half_w:
                break

            half_w = new_half_w
            result = _fit_window(px, py, half_w)
            if result is None:
                break

        if result is None:
            continue

        mean_x, mean_y, cov, sum_p = result
        gaussians.append(
            {
                "mean": np.array([mean_x, mean_y]),
                "global_peak": global_peak,
                "peak": np.array([float(px), float(py)]),
                "cov": cov,
                "weight": sum_p,
                "half_w": half_w,
            }
        )

    return gaussians


def extract_gaussians_from_heatmap2(heatmap, threshold=0.1, window_size=5):
    """
    Finds local peaks in a probability heatmap and fits a 2D Gaussian
    (mean and covariance) to the local neighborhood around each peak
    using high-speed Image Moments.
    """
    if heatmap.dtype != np.float32:
        heatmap = heatmap.astype(np.float32)

    # global peak for ransac initialization (falls back to local peaks if key is absent for old data)
    global_peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    global_peak = np.array([float(global_peak_idx[1]), float(global_peak_idx[0])])  # (x, y) format

    # 1. Gaussian blurring to smooth the heatmap and reduce noise
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)

    # 2. Find local maxima (cv2.dilate is identical to maximum_filter here but faster)
    kernel = np.ones((window_size, window_size), np.uint8)
    local_max = cv2.dilate(heatmap, kernel)

    # 3. Mask for peaks
    peaks = (heatmap > threshold) & (local_max == heatmap)
    y_peaks, x_peaks = np.where(peaks)

    gaussians = []
    half_w = window_size // 2
    h, w = heatmap.shape

    for px, py in zip(x_peaks, y_peaks):
        y_min = max(0, py - half_w)
        y_max = min(h, py + half_w + 1)
        x_min = max(0, px - half_w)
        x_max = min(w, px + half_w + 1)

        window = heatmap[y_min:y_max, x_min:x_max]
        M = cv2.moments(window)
        sum_p = M["m00"]
        if sum_p == 0:
            continue

        local_mean_x = M["m10"] / sum_p
        local_mean_y = M["m01"] / sum_p
        mean_x = x_min + local_mean_x
        mean_y = y_min + local_mean_y

        cov_xx = M["mu20"] / sum_p
        cov_yy = M["mu02"] / sum_p
        cov_xy = M["mu11"] / sum_p

        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
        cov += np.eye(2) * 1e-4

        gaussians.append(
            {
                "mean": np.array([mean_x, mean_y]),
                "cov": cov,
                "weight": sum_p,
                "peaks": np.array([float(px), float(py)]),
                "global_peak": global_peak,
            }
        )

    return gaussians
