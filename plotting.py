import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap_comparison(heatmap, gaussians, px, py):
    """
    Reconstruct a heatmap from fitted Gaussians and display a side-by-side
    comparison with the original heatmap.
    """
    h, w = heatmap.shape
    ys, xs = np.mgrid[0:h, 0:w]
    pos = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)
    reconstructed = np.zeros(h * w, dtype=np.float64)

    for g in gaussians:
        mean = np.array(g["mean"], dtype=np.float64)
        cov = np.array(g["cov"], dtype=np.float64)
        weight = float(g["weight"])
        try:
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)
            diff = pos - mean
            exponent = -0.5 * np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
            norm = weight / (2 * np.pi * np.sqrt(max(cov_det, 1e-12)))
            reconstructed += norm * np.exp(exponent)
        except np.linalg.LinAlgError:
            pass

    reconstructed = reconstructed.reshape(h, w)

    fig_cmp, (ax_orig, ax_rec) = plt.subplots(1, 2, figsize=(10, 4))
    ax_orig.imshow(heatmap, cmap="hot", interpolation="nearest", origin="upper")
    ax_orig.set_title(f"Original heatmap  (patch {px},{py})")
    im_rec = ax_rec.imshow(reconstructed, cmap="hot", interpolation="nearest", origin="upper")
    ax_rec.set_title("Recreated (Gaussian mixture)")
    plt.colorbar(im_rec, ax=ax_rec)
    plt.tight_layout()
    plt.show()


def plot_homography_projection(pts_A, means_B, H, size_a=14, size_b=56, title_suffix="", save_path=None):
    """
    Plot image A alongside its projection into image B space via homography H.
    """
    img_A = np.zeros((size_a, size_a, 3), dtype=np.uint8)
    for r in range(size_a):
        for c in range(size_a):
            img_A[r, c] = [int(c / size_a * 255), int(r / size_a * 255), 128]

    if H is not None:
        img_A_in_B = cv2.warpPerspective(
            img_A,
            H,
            (size_b, size_b),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    else:
        img_A_in_B = np.zeros((size_b, size_b, 3), dtype=np.uint8)
        print("Warning: H is None, cannot project.")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 5))
    ax_a.imshow(
        img_A,
        interpolation="nearest",
        origin="upper",
        extent=[-0.5, size_a - 0.5, size_a - 0.5, -0.5],
    )
    ax_a.set_xticks(np.arange(-0.5, size_a, 1), minor=True)
    ax_a.set_yticks(np.arange(-0.5, size_a, 1), minor=True)
    ax_a.grid(which="minor", color="white", linewidth=0.5)
    ax_a.set_title(f"Image A  ({size_a}×{size_a}) {title_suffix}")
    ax_a.set_xlabel("x (patch index)")
    ax_a.set_ylabel("y (patch index)")
    if len(pts_A):
        ax_a.scatter(pts_A[:, 0], pts_A[:, 1], s=8, c="cyan", zorder=5, label="pts A")
        ax_a.legend(fontsize=8)

    ax_b.imshow(
        img_A_in_B,
        interpolation="nearest",
        origin="upper",
        extent=[-0.5, size_b - 0.5, size_b - 0.5, -0.5],
    )
    ax_b.set_xticks(np.arange(-0.5, size_b, 4), minor=True)
    ax_b.set_yticks(np.arange(-0.5, size_b, 4), minor=True)
    ax_b.grid(which="minor", color="white", linewidth=0.3, alpha=0.5)
    ax_b.set_title(f"Image A projected into Image B space  ({size_b}×{size_b}) {title_suffix}")
    ax_b.set_xlabel("x (pixels in B)")
    ax_b.set_ylabel("y (pixels in B)")
    if len(means_B):
        ax_b.scatter(means_B[:, 0], means_B[:, 1], s=8, c="red", zorder=5, label="Gaussian means B")
        ax_b.legend(fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_correspondences_with_arrows(pts_A, means_B, peaks_B, covs_B, size_a, size_b, save_path=None):
    """
    Visualize pts_A on the left, means_B, peaks_B, covs_B on the right,
    with arrows connecting correspondences from the left image to the right image.
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 5))

    ax_a.set_xlim(-0.5, size_a - 0.5)
    ax_a.set_ylim(size_a - 0.5, -0.5)
    ax_a.set_xticks(np.arange(-0.5, size_a, 1), minor=True)
    ax_a.set_yticks(np.arange(-0.5, size_a, 1), minor=True)
    ax_a.grid(which="minor", color="black", linewidth=0.5, alpha=0.1)
    ax_a.set_title("Image A (pts_A)")
    ax_a.set_xlabel("x (patch index)")
    ax_a.set_ylabel("y (patch index)")
    ax_a.scatter(pts_A[:, 0], pts_A[:, 1], s=15, c="cyan", zorder=5, label="pts A")
    ax_a.legend()

    ax_b.set_xlim(-0.5, size_b - 0.5)
    ax_b.set_ylim(size_b - 0.5, -0.5)
    ax_b.set_xticks(np.arange(-0.5, size_b, 4), minor=True)
    ax_b.set_yticks(np.arange(-0.5, size_b, 4), minor=True)
    ax_b.grid(which="minor", color="black", linewidth=0.3, alpha=0.1)
    ax_b.set_title("Image B (Gaussians)")
    ax_b.set_xlabel("x (pixels in B)")
    ax_b.set_ylabel("y (pixels in B)")
    ax_b.scatter(peaks_B[:, 0], peaks_B[:, 1], c="orange", s=20, marker="x", label="peaks B", zorder=4)
    ax_b.scatter(means_B[:, 0], means_B[:, 1], c="red", s=15, label="means B", zorder=5)

    from matplotlib.patches import Ellipse

    for mean, cov in zip(means_B, covs_B):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = np.maximum(vals[order], 1e-6)
        vecs = vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 3 * np.sqrt(vals)
        ellip = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=theta,
            edgecolor="red",
            facecolor="none",
            alpha=0.3,
            zorder=3,
        )
        ax_b.add_patch(ellip)

    ax_b.legend()

    for pt_a, mean_b in zip(pts_A, means_B):
        con = patches.ConnectionPatch(
            xyA=pt_a,
            xyB=mean_b,
            coordsA="data",
            coordsB="data",
            axesA=ax_a,
            axesB=ax_b,
            color="black",
            alpha=0.3,
            arrowstyle="-|>",
            mutation_scale=10,
        )
        ax_b.add_artist(con)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_image_homography_warp(im_A, im_B, H):
    """
    Overlays im_A onto im_B using homography H, drawing a red border around the warped im_A.
    """
    import torch

    if isinstance(im_A, torch.Tensor):
        img_a = im_A.permute(1, 2, 0).cpu().numpy()
    else:
        img_a = im_A
    if isinstance(im_B, torch.Tensor):
        img_b = im_B.permute(1, 2, 0).cpu().numpy()
    else:
        img_b = im_B

    img_a = (img_a - img_a.min()) / (img_a.max() - img_a.min() + 1e-8)
    img_b = (img_b - img_b.min()) / (img_b.max() - img_b.min() + 1e-8)

    h_b, w_b = img_b.shape[:2]
    h_a, w_a = img_a.shape[:2]

    warped_A = cv2.warpPerspective(img_a, H, (w_b, h_b))

    mask = cv2.warpPerspective(np.ones_like(img_a, dtype=np.float32), H, (w_b, h_b))
    mask_binary = (np.sum(mask, axis=2, keepdims=True) > 0).astype(np.float32)

    blended = img_b * (1 - mask_binary) + warped_A * mask_binary

    plt.figure(figsize=(10, 10))
    plt.imshow(blended)

    corners_A = np.array([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]], dtype=np.float32).reshape(-1, 1, 2)
    corners_B = cv2.perspectiveTransform(corners_A, H).reshape(-1, 2)
    polygon = np.vstack((corners_B, corners_B[0]))
    plt.plot(polygon[:, 0], polygon[:, 1], color="red", linewidth=1.5)

    plt.axis("off")
    plt.title("Image A mapped onto Image B")
    plt.tight_layout()
