"""
Polynomial fitting for sub-pixel corner refinement.

Uses cone-shaped spatial weighting to suppress outlier influence, then
fits quadratic (saddle) or cubic (monkey-saddle) surfaces iteratively.
Based on ROCHEDE (ECCV 2014) and Deltille (ICCV 2017) papers.
"""

import math

import numpy as np

from pycbdetect.config import Corner, CornerType, Params
from pycbdetect.utils import get_image_patch_with_mask


def _cone_filter_kernel(r):
    """Build cone-shaped averaging kernel and boolean mask.

    Weight decreases linearly with radial distance from center.
    Mask marks elements strictly inside the cone (dist < r+1).
    """
    sz = 2 * r + 1
    kernel = np.zeros((sz, sz), dtype=np.float64)
    nz_count = 0
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            d = math.sqrt(i * i + j * j)
            val = max(0.0, r + 1 - d)
            kernel[i + r, j + r] = val
            if val < 1e-6:
                nz_count += 1
    s = kernel.sum()
    if s > 1e-10:
        kernel /= s
    return kernel, nz_count


def _build_design_matrix(r, nz_count, degree_order):
    """Build Vandermonde-like design matrix for surface fitting.

    degree_order=2 → Ax² + Ay² + Axy + Bx + Cy + D  (6 coefs)
    degree_order=3 → Ax³ + Ax²y + Axy² + Ay³ + Bx² + Bxy + By² + Cx + Dy + E  (10 coefs)
    """
    sz = 2 * r + 1
    mask_inner = np.zeros((sz, sz), dtype=bool)
    nz_local = 0
    for i in range(sz):
        for j in range(sz):
            d = math.sqrt((i - r) ** 2 + (j - r) ** 2)
            if d < r + 1:
                mask_inner[i, j] = True
            else:
                nz_local += 1

    assert nz_local == nz_count

    ncols = 10 if degree_order == 3 else 6
    nrows = sz * sz - nz_count
    A = np.zeros((nrows, ncols), dtype=np.float64)
    row = 0
    for j in range(-r, r + 1):
        for i in range(-r, r + 1):
            if not mask_inner[j + r, i + r]:
                continue
            if degree_order == 3:
                A[row, 0] = i * i * i
                A[row, 1] = i * i * j
                A[row, 2] = i * j * j
                A[row, 3] = j * j * j
                A[row, 4] = i * i
                A[row, 5] = i * j
                A[row, 6] = j * j
                A[row, 7] = i
                A[row, 8] = j
                A[row, 9] = 1
            else:
                A[row, 0] = i * i
                A[row, 1] = j * j
                A[row, 2] = i * j
                A[row, 3] = i
                A[row, 4] = j
                A[row, 5] = 1
            row += 1

    pinv_atb = np.linalg.lstsq(A.T @ A, A.T, rcond=-1)[0]
    return pinv_atb, mask_inner


def _gaussian_blur_convolve(img, kernel):
    """Simple 2D convolution with replicate padding."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 1 // 2
    pw = kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
    oh, ow = img.shape
    result = np.zeros((oh, ow), dtype=np.float64)
    for j in range(ph, ph + oh):
        for i in range(pw, pw + ow):
            result[j - ph, i - pw] = np.sum(
                padded[j - ph:j + ph + 1, i - pw:i + pw + 1] * kernel
            )
    return result


def polynomial_fit_saddle(img, r, corners):
    """Quadratic surface fit for saddle-point corners.

    Fits z = k0·x² + k1·y² + k2·xy + k3·x + k4·y + k5
    and relocates corner to the stationary point ∂z/∂x = ∂z/∂y = 0.
    """
    max_iter = 5
    eps = 0.01
    h, w = img.shape

    ker, nz = _cone_filter_kernel(r)
    blurred = _gaussian_blur_convolve(img, ker)

    pinv_atb, mask_inner = _build_design_matrix(r, nz, 2)

    keep = np.zeros(len(corners.p), dtype=bool)

    for ci in range(len(corners.p)):
        u_cur, v_cur = corners.p[ci][0], corners.p[ci][1]
        ok = True

        for _nit in range(max_iter):
            if u_cur - r < 0 or u_cur + r >= w - 1 or v_cur - r < 0 or v_cur + r >= h - 1:
                ok = False
                break

            b = get_image_patch_with_mask(blurred, mask_inner, u_cur, v_cur, r)
            k = pinv_atb @ b

            # Determinant of Hessian of fitted quadric
            det = 4 * k[0, 0] * k[1, 0] - k[2, 0] ** 2
            if det > 0:
                ok = False
                break

            dx = (k[2, 0] * k[4, 0] - 2 * k[1, 0] * k[3, 0]) / det
            dy = (k[2, 0] * k[3, 0] - 2 * k[0, 0] * k[4, 0]) / det

            u_new, v_new = u_cur + dx, v_cur + dy
            disp = math.sqrt((u_new - corners.p[ci][0]) ** 2 + (v_new - corners.p[ci][1]) ** 2)
            if disp > r:
                ok = False
                break
            if math.sqrt(dx ** 2 + dy ** 2) <= eps:
                u_cur, v_cur = u_new, v_new
                break
            u_cur, v_cur = u_new, v_new

        if ok:
            keep[ci] = True
            corners.p[ci] = np.array([u_cur, v_cur], dtype=np.float64)

    fp, fr, fv1, fv2 = [], [], [], []
    for ci in range(len(corners.p)):
        if keep[ci]:
            fp.append(corners.p[ci].copy())
            fr.append(corners.r[ci])
            fv1.append(corners.v1[ci].copy())
            fv2.append(corners.v2[ci].copy())

    corners.p = fp
    corners.r = fr
    corners.v1 = fv1
    corners.v2 = fv2


def polynomial_fit_monkey_saddle(img, r, corners):
    """Cubic surface fit for monkey-saddle corners.

    Fits z = Σ_{deg≤3} coeff · monomial(x,y)
    and solves for the degenerate critical point.
    """
    max_iter = 5
    eps = 0.001
    h, w = img.shape

    ker, nz = _cone_filter_kernel(r)
    blurred = _gaussian_blur_convolve(img, ker)

    pinv_atb, mask_inner = _build_design_matrix(r, nz, 3)

    keep = np.zeros(len(corners.p), dtype=bool)

    for ci in range(len(corners.p)):
        u_cur, v_cur = corners.p[ci][0], corners.p[ci][1]
        ok = True

        for _nit in range(max_iter):
            if u_cur - r < 0 or u_cur + r >= w - 1 or v_cur - r < 0 or v_cur + r >= h - 1:
                ok = False
                break

            b = get_image_patch_with_mask(blurred, mask_inner, u_cur, v_cur, r)
            k = pinv_atb @ b

            # Monkey-saddle condition
            det = 3 * (k[0, 0] * k[2, 0] + k[1, 0] * k[3, 0]) - (k[1, 0] ** 2 + k[2, 0] ** 2)
            if det > 0:
                ok = False
                break

            # Solve for displacement from second-order stationarity
            Ta = np.array([
                [3.0 * k[0, 0], k[1, 0]],
                [2.0 * k[1, 0], 2.0 * k[2, 0]],
                [k[2, 0], 3.0 * k[3, 0]],
            ], dtype=np.float64)
            Tb = np.array([[-k[4, 0]], [-k[5, 0]], [-k[6, 0]]], dtype=np.float64)
            Tx, _resid, _, _ = np.linalg.lstsq(Ta, Tb, rcond=-1)

            dx, dy = Tx[0, 0], Tx[1, 0]
            u_new, v_new = u_cur + dx, v_cur + dy
            disp = math.sqrt((u_new - corners.p[ci][0]) ** 2 + (v_new - corners.p[ci][1]) ** 2)
            if disp > r:
                ok = False
                break
            if math.sqrt(dx ** 2 + dy ** 2) <= eps:
                u_cur, v_cur = u_new, v_new
                break
            u_cur, v_cur = u_new, v_new

        if ok:
            keep[ci] = True
            corners.p[ci] = np.array([u_cur, v_cur], dtype=np.float64)

    fp, fr, fv1, fv2, fv3 = [], [], [], [], []
    for ci in range(len(corners.p)):
        if keep[ci]:
            fp.append(corners.p[ci].copy())
            fr.append(corners.r[ci])
            fv1.append(corners.v1[ci].copy())
            fv2.append(corners.v2[ci].copy())
            fv3.append(corners.v3[ci].copy())

    corners.p = fp
    corners.r = fr
    corners.v1 = fv1
    corners.v2 = fv2
    corners.v3 = fv3


def polynomial_fit(img, corners, params):
    """Dispatch to saddle or monkey-saddle variant."""
    if params.corner_type == CornerType.SaddlePoint:
        polynomial_fit_saddle(img, params.polynomial_fit_half_kernel_size, corners)
    else:
        polynomial_fit_monkey_saddle(img, params.polynomial_fit_half_kernel_size, corners)
