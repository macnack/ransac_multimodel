import numpy as np


def convert_to_pixel_homography(
    H_feat,
    in_patch_dim,
    out_patch_dim,
    crop_res=(224, 224),
    map_res=(896, 896),
):
    """
    Converts a homography estimated in feature map grid space to a homography
    mapping from Image A pixels to Image B pixels.
    """
    H_A, W_A = crop_res
    H_B, W_B = map_res

    scale_A_x = in_patch_dim / W_A
    scale_A_y = in_patch_dim / H_A
    scale_B_x = out_patch_dim / W_B
    scale_B_y = out_patch_dim / H_B

    S_A = np.array([[scale_A_x, 0, 0], [0, scale_A_y, 0], [0, 0, 1]])
    S_B = np.array([[scale_B_x, 0, 0], [0, scale_B_y, 0], [0, 0, 1]])

    H_A_to_B = np.linalg.inv(S_B) @ H_feat @ S_A
    return H_A_to_B / H_A_to_B[2, 2]


def convert_to_dataloader_homography(
    H_feat,
    in_patch_dim,
    out_patch_dim,
    crop_res=(256, 256),
    map_res=(1024, 1024),
):
    """
    Converts a homography estimated in feature map grid space back to the
    original dataloader's ground-truth homography coordinate space.
    """
    H_A, W_A = crop_res
    H_B, W_B = map_res

    offset_x = (W_B - W_A) / 2
    offset_y = (H_B - H_A) / 2

    scale_A_x = in_patch_dim / W_A
    scale_A_y = in_patch_dim / H_A
    scale_B_x = out_patch_dim / W_B
    scale_B_y = out_patch_dim / H_B

    S_A = np.array([[scale_A_x, 0, 0], [0, scale_A_y, 0], [0, 0, 1]])
    S_B = np.array([[scale_B_x, 0, 0], [0, scale_B_y, 0], [0, 0, 1]])
    T_offset = np.array([[1, 0, -offset_x], [0, 1, -offset_y], [0, 0, 1]])

    H_inv = np.linalg.inv(S_B) @ H_feat @ S_A @ T_offset
    H_dl = np.linalg.inv(H_inv)
    H_dl = H_dl / H_dl[2, 2]
    return H_dl
