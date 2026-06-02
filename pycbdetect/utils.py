"""
Utility functions ported from various C++ source files.

Includes: weight_mask, create_correlation_patch, get_image_patch,
hessian_response, box_filter, arctan2 computation.
"""

import math

import numpy as np
from scipy.ndimage import uniform_filter


def weight_mask(radius_list):
    """Compute annular weight masks for given radii.

    Returns dict mapping radius -> 2D array of weights.
    Pixels near the ring at fraction 0.7..1.3 of radius get higher weight.
    """
    mask = {}
    for r in radius_list:
        size = 2 * r + 1
        arr = np.zeros((size, size), dtype=np.float64)
        for v in range(size):
            for u in range(size):
                dist = math.sqrt((u - r) ** 2 + (v - r) ** 2) / r
                dist = min(max(dist, 0.7), 1.3)
                arr[v, u] = (1.3 - dist) / 0.6
        mask[r] = arr
    return mask


def create_correlation_patch_2angle(angle_1, angle_2, radius):
    """Create 4 correlation patches for 2-angle (saddle point) case.

    Returns list of 4 kernels: [a1, a2, b1, b2].
    """
    width = 2 * radius + 1
    height = 2 * radius + 1
    kernels = [np.zeros((height, width), dtype=np.float64) for _ in range(4)]

    mu = radius + 1
    mv = radius + 1

    # Compute normals from angles
    n1 = np.array([-math.sin(angle_1), math.cos(angle_1)])
    n2 = np.array([-math.sin(angle_2), math.cos(angle_2)])

    for u in range(width):
        for v in range(height):
            vec = np.array([u + 1 - mu, v + 1 - mv], dtype=np.int64)
            dist = math.sqrt(float(vec[0] ** 2 + vec[1] ** 2))

            s1 = float(np.dot(vec, n1))
            s2 = float(np.dot(vec, n2))

            if dist <= radius:
                if s1 <= -0.1 and s2 <= -0.1:
                    kernels[0][v, u] = 1.0
                elif s1 >= 0.1 and s2 >= 0.1:
                    kernels[1][v, u] = 1.0
                elif s1 <= -0.1 and s2 >= 0.1:
                    kernels[2][v, u] = 1.0
                elif s1 >= 0.1 and s2 <= -0.1:
                    kernels[3][v, u] = 1.0

    # Normalize
    for k in kernels:
        s = np.sum(k)
        if s > 1e-5:
            k /= s

    return kernels


def create_correlation_patch_3angle(angle_1, angle_2, angle_3, radius):
    """Create 6 correlation patches for 3-angle (monkey saddle) case.

    Returns list of 6 kernels: [a1, a2, a3, b1, b2, b3].
    """
    width = 2 * radius + 1
    height = 2 * radius + 1
    kernels = [np.zeros((height, width), dtype=np.float64) for _ in range(6)]

    mu = radius + 1
    mv = radius + 1

    n1 = np.array([-math.sin(angle_1), math.cos(angle_1)])
    n2 = np.array([-math.sin(angle_2), math.cos(angle_2)])
    n3 = np.array([-math.sin(angle_3), math.cos(angle_3)])

    for u in range(width):
        for v in range(height):
            vec = np.array([u + 1 - mu, v + 1 - mv], dtype=np.int64)
            dist = math.sqrt(float(vec[0] ** 2 + vec[1] ** 2))

            s1 = float(np.dot(vec, n1))
            s2 = float(np.dot(vec, n2))
            s3 = float(np.dot(vec, n3))

            if dist <= radius:
                if s1 >= -0.1 and s2 <= -0.1:
                    kernels[0][v, u] = 1.0
                elif s1 >= 0.1 and s3 >= 0.1:
                    kernels[1][v, u] = 1.0
                elif s2 <= -0.1 and s3 >= 0.1:
                    kernels[2][v, u] = 1.0
                elif s1 <= 0.1 and s2 >= -0.1:
                    kernels[3][v, u] = 1.0
                elif s1 <= 0.1 and s3 <= -0.1:
                    kernels[4][v, u] = 1.0
                elif s2 >= 0.1 and s3 <= -0.1:
                    kernels[5][v, u] = 1.0

    for k in kernels:
        s = np.sum(k)
        if s > 1e-5:
            k /= s

    return kernels


def get_image_patch(img, u, v, r):
    """Extract bilinear-interpolated image patch centered at (u,v) with half-size r.

    Args:
        img: 2D numpy array (float64)
        u, v: floating-point center coordinates
        r: integer half-width/height of patch

    Returns:
        2D numpy array of shape (2*r+1, 2*r+1)
    """
    iu = int(round(u))
    iv = int(round(v))
    du = u - iu
    dv = v - iv
    a00 = 1 - du - dv + du * dv
    a01 = du - du * dv
    a10 = dv - du * dv
    a11 = du * dv

    patch = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float64)
    for j in range(-r, r + 1):
        for i in range(-r, r + 1):
            patch[j + r, i + r] = (
                a00 * img[iv + j, iu + i]
                + a01 * img[iv + j, iu + i + 1]
                + a10 * img[iv + j + 1, iu + i]
                + a11 * img[iv + j + 1, iu + i + 1]
            )
    return patch


def get_image_patch_with_mask(img, mask, u, v, r):
    """Extract masked bilinear-interpolated patch as flat column vector.

    Only includes pixels where mask >= 1e-6.
    """
    iu = int(round(u))
    iv = int(round(v))
    du = u - iu
    dv = v - iv
    a00 = 1 - du - dv + du * dv
    a01 = du - du * dv
    a10 = dv - du * dv
    a11 = du * dv

    vals = []
    for j in range(-r, r + 1):
        for i in range(-r, r + 1):
            if mask[j + r, i + r] >= 1e-6:
                vals.append(
                    a00 * img[iv + j, iu + i]
                    + a01 * img[iv + j, iu + i + 1]
                    + a10 * img[iv + j + 1, iu + i]
                    + a11 * img[iv + j + 1, iu + i + 1]
                )
    return np.array(vals, dtype=np.float64).reshape(-1, 1)


def box_filter(img, kernel_radius):
    """Box (uniform) filter with boundary handling identical to C++ version.

    Port of box_filter() from image_normalization_and_gradients.cc.
    Uses a separable approach: vertical sliding window accumulated in buf[],
    then horizontal sliding window applied via delayed-division strategy
    directly on the result buffer.

    Args:
        img: 2D numpy array (H x W, float64)
        kernel_radius: half-kernel size (used for both X and Y direction)

    Returns:
        Blurred 2D array.
    """
    h, w = img.shape
    ky = kernel_radius
    kx = kernel_radius

    buf = np.zeros(w, dtype=np.float64)
    count_buf = np.zeros(w, dtype=np.int64)

    # Vertical initialization: accumulate rows 0..min(ky, h-2)
    for j in range(min(ky, h - 1)):
        buf += img[j]
        count_buf += 1

    result = np.zeros_like(img)
    for j in range(h):
        # Slide vertical window
        if j > ky:
            buf -= img[j - ky - 1]
            count_buf -= 1
        if j + ky < h:
            buf += img[j + ky]
            count_buf += 1

        # Horizontal scan – delayed division (identical to C++)
        # Step 1: compute sum for column 0
        count = 0
        for i in range(min(kx, w - 1) + 1):
            result[j, 0] += buf[i]
            count += count_buf[i]

        # Step 2: slide horizontally, storing raw sums and dividing later
        for i in range(1, w):
            result[j, i] = result[j, i - 1]
            result[j, i - 1] /= count if count != 0 else 1.0
            if i > kx:
                result[j, i] -= buf[i - kx - 1]
                count -= count_buf[i - kx - 1]
            if i + kx < w:
                result[j, i] += buf[i + kx]
                count += count_buf[i + kx]

        # Last column needs explicit division
        result[j, w - 1] /= count if count != 0 else 1.0

    return result


def hessian_response(img):
    """Compute determinant of Hessian response map.

    Uses central difference approximations for Lxx, Lyx, Lyy.
    """
    h, w = img.shape
    result = np.zeros_like(img)

    for i in range(1, h - 1):
        for c in range(1, w - 1):
            v11 = img[i - 1, c - 1]
            v12 = img[i - 1, c]
            v13 = img[i - 1, c + 1]
            v21 = img[i, c - 1]
            v22 = img[i, c]
            v23 = img[i, c + 1]
            v31 = img[i + 1, c - 1]
            v32 = img[i + 1, c]
            v33 = img[i + 1, c + 1]

            Lxx = v21 - 2.0 * v22 + v23
            Lyy = v12 - 2.0 * v22 + v32
            Lxy = (v13 - v11 + v31 - v33) / 4.0

            result[i, c] = Lxx * Lyy - Lxy * Lxy

    return result


def atan2_safe(y, x):
    """Vectorized safe atan2 returning values in (-pi, pi]."""
    return np.arctan2(y, x)


def convolve2d_border_replicate(img, kernel):
    """Convolve 2D image with kernel using replicate padding (border reflect)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # Pad with reflect
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')

    oh, ow = img.shape
    output = np.zeros((oh, ow), dtype=np.float64)

    for j in range(ph, ph + oh):
        for i in range(pw, pw + ow):
            region = padded[j - ph:j + ph + 1, i - pw:i + pw + 1]
            output[j - ph, i - pw] = np.sum(region * kernel)

    return output
