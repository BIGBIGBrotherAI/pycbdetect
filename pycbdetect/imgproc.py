"""
Image normalization and gradient computation.

Computes Sobel-derived horizontal/vertical gradients, gradient magnitude,
and gradient angle maps.
"""

import math

import numpy as np

from pycbdetect.utils import box_filter, convolve2d_border_replicate


def image_normalization_and_gradients(img, params):
    """Normalize image and compute gradient info.

    Args:
        img: Grayscale image normalized to [0,1], float64
        params: Params object

    Returns:
        tuple: (img_normalized, img_du, img_dv, img_angle, img_weight)
    """
    # Normalize image
    if params.norm:
        blur_img = box_filter(img.copy(), params.norm_half_kernel_size)
        img = img.astype(np.float64) - blur_img
        img = 2.5 * np.clip(img + 0.2, 0, 0.4) - 0.5
        img = np.maximum(img, 0)

    # Sobel kernels
    du_kern = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
    dv_kern = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

    # Compute image derivatives
    img_du = convolve2d_border_replicate(img, du_kern)
    img_dv = convolve2d_border_replicate(img, dv_kern)

    # Gradient angle
    img_angle = np.arctan2(img_dv, img_du)
    img_angle = np.where(img_angle >= math.pi, img_angle - math.pi, img_angle)

    # Gradient magnitude (= weight)
    img_weight = np.sqrt(img_du ** 2 + img_dv ** 2)

    # Scale input image to [0, 1]
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 1e-10:
        img = (img - img_min) / (img_max - img_min)

    return img, img_du, img_dv, img_angle, img_weight
