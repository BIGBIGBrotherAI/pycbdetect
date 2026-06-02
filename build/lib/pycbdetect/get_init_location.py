"""
Get initial corner locations using various detection strategies.

Implements TemplateMatching (fast/slow), Hessian Response, and
Localized Radon Transform approaches.
"""

import math

import numpy as np
from scipy.signal import fftconvolve

from pycbdetect.config import Corner, CornerType, DetectMethod, Params
from pycbdetect.nms import non_maximum_suppression
from pycbdetect.utils import (
    create_correlation_patch_2angle,
    get_image_patch,
    hessian_response,
)


def _rotate_image(img, angle, out_size=None):
    """Rotate image by angle radians around center, preserving geometry."""
    if abs(angle) < 1e-3:
        return img.copy()

    h, w = img.shape
    cu, cv = (w - 1) / 2.0, (h - 1) / 2.0

    ca, sa = math.cos(angle), math.sin(angle)

    if out_size is None:
        # Calculate bounding box of rotated rectangle
        pts = [
            (-cu * ca - cv * sa, cu * sa - cv * ca),
            (cu * ca - cv * sa, -cu * sa - cv * ca),
            (-cu * ca + cv * sa, cu * sa + cv * ca),
            (cu * ca + cv * sa, -cu * sa + cv * ca),
        ]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        nw = int(round(max(xs) - min(xs))) + 1
        nh = int(round(max(ys) - min(ys))) + 1
    else:
        nw, nh = out_size

    ou, ov = (nw - 1) / 2.0, (nh - 1) / 2.0
    shift_u = ou - cu * ca - cv * sa
    shift_v = ov + cu * sa - cv * ca

    # Build affine transform matrix and apply warp
    # We manually interpolate for correctness
    result = np.zeros((nh, nw), dtype=np.float64)
    for vy in range(nh):
        for ux in range(nw):
            # Map destination to source
            sx = (ux - ou) * ca - (vy - ov) * sa + cu
            sy = (ux - ou) * sa + (vy - ov) * ca + cv

            ix = int(round(sx))
            iy = int(round(sy))
            du = sx - ix
            dv = sy - iy

            if 0 <= ix < w - 1 and 0 <= iy < h - 1:
                a00 = 1 - du - dv + du * dv
                a01 = du - du * dv
                a10 = dv - du * dv
                a11 = du * dv
                result[vy, ux] = (
                    a00 * img[iy, ix]
                    + a01 * img[iy, ix + 1]
                    + a10 * img[iy + 1, ix]
                    + a11 * img[iy + 1, ix + 1]
                )

    return result


def _localized_radon_transform(img):
    """Accurate Detection and Localization of Checkerboard Corners for Calibration.

    Rotates image, applies separable blurs, rotates back, computes contrast.
    """
    angles = [0.0, math.pi / 4]
    rh, rw = img.shape

    rb_imgs = []
    for ang in angles:
        r_img = _rotate_image(img, -ang)
        # Blur horizontally
        u_img = np.zeros_like(r_img)
        for y in range(r_img.shape[0]):
            for x in range(r_img.shape[1]):
                lo = max(0, x - 5)
                hi = min(r_img.shape[1], x + 6)
                u_img[y, x] = np.mean(r_img[y, lo:hi])
        # Blur vertically
        v_img = np.zeros_like(r_img)
        for y in range(r_img.shape[0]):
            for x in range(r_img.shape[1]):
                lo = max(0, y - 5)
                hi = min(r_img.shape[0], y + 6)
                v_img[y, x] = np.mean(r_img[lo:hi, x])

        ru = _rotate_image(u_img, ang, (rw, rh))
        rv = _rotate_image(v_img, ang, (rw, rh))
        rb_imgs.extend([ru, rv])

    max_img_1 = np.maximum(rb_imgs[0], rb_imgs[1])
    max_img_2 = np.maximum(rb_imgs[2], rb_imgs[3])
    min_img_1 = np.minimum(rb_imgs[0], rb_imgs[1])
    min_img_2 = np.minimum(rb_imgs[2], rb_imgs[3])
    result = np.maximum(max_img_1, max_img_2) - np.minimum(min_img_1, min_img_2)
    result = result ** 2
    return result


def _location_refinement(img, img_du, img_dv, corners):
    """Sub-pixel refinement of corner positions using gradient-based optimization."""
    h, w = img.shape[:2]
    for i in range(len(corners.p)):
        u = corners.p[i][0]
        v = corners.p[i][1]
        r = corners.r[i]

        if u - r < 0 or u + r >= w - 1 or v - r < 0 or v + r >= h - 1:
            continue

        img_du_sub = get_image_patch(img_du, u, v, r)
        img_dv_sub = get_image_patch(img_dv, u, v, r)

        G = np.zeros((2, 2), dtype=np.float64)
        b = np.zeros((2,), dtype=np.float64)

        for j2 in range(2 * r + 1):
            for i2 in range(2 * r + 1):
                o_du = img_du_sub[j2, i2]
                o_dv = img_dv_sub[j2, i2]
                o_norm = math.sqrt(o_du ** 2 + o_dv ** 2)
                if o_norm < 0.1:
                    continue
                if i2 == r and j2 == r:
                    continue
                G[0, 0] += o_du * o_du
                G[0, 1] += o_du * o_dv
                G[1, 0] += o_du * o_dv
                G[1, 1] += o_dv * o_dv
                b[0] += o_du * o_du * (i2 - r + u) + o_du * o_dv * (j2 - r + v)
                b[1] += o_du * o_dv * (i2 - r + u) + o_dv * o_dv * (j2 - r + v)

        try:
            new_pos = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            continue

        if (abs(new_pos[0] - corners.p[i][0]) + abs(new_pos[1] - corners.p[i][1])
                < corners.r[i] * 2):
            corners.p[i] = np.array(new_pos, dtype=np.float64)


def get_init_location(img, img_du, img_dv, corners, params):
    """Main entry: detect initial corner locations.

    Dispatches to strategy based on params.detect_method and params.corner_type.
    Appends found corners to the ``corners`` Container.
    """
    detect_method = params.detect_method
    if params.corner_type == CornerType.MonkeySaddlePoint:
        detect_method = DetectMethod.HessianResponse

    if detect_method in (DetectMethod.TemplateMatchFast, DetectMethod.TemplateMatchSlow):
        # Angle pairs
        if detect_method == DetectMethod.TemplateMatchFast:
            tprops = [0, math.pi / 2, math.pi / 4, -math.pi / 4]
        else:
            tprops = [
                0, math.pi / 2,
                math.pi / 4, -math.pi / 4,
                0, math.pi / 4,
                0, -math.pi / 4,
                math.pi / 4, math.pi / 2,
                -math.pi / 4, math.pi / 2,
                -3 * math.pi / 8, 3 * math.pi / 8,
                -math.pi / 8, math.pi / 8,
                -math.pi / 8, -3 * math.pi / 8,
                math.pi / 8, 3 * math.pi / 8,
            ]

        for r in params.radius:
            img_corners = np.zeros_like(img)

            for ki in range(0, len(tprops), 2):
                kernels = create_correlation_patch_2angle(tprops[ki], tprops[ki + 1], r)

                # Convolve
                imgs = []
                for kern in kernels:
                    imgs.append(convolve2d_border_replicate_fast(img, kern))

                img_corners_a1, img_corners_a2, img_corners_b1, img_corners_b2 = imgs

                img_corners_mu = (img_corners_a1 + img_corners_a2 + img_corners_b1 + img_corners_b2) / 4.0

                # Case 1: a=white, b=black
                img_corners_a = np.minimum(img_corners_a1, img_corners_a2) - img_corners_mu
                img_corners_b = img_corners_mu - np.maximum(img_corners_b1, img_corners_b2)
                img_corners_s1 = np.minimum(img_corners_a, img_corners_b)

                # Case 2: b=white, a=black
                img_corners_a = img_corners_mu - np.maximum(img_corners_a1, img_corners_a2)
                img_corners_b = img_corners_b1.min(axis=(0, 1))  # placeholder – simplified
                img_corners_b = np.minimum(img_corners_b1, img_corners_b2) - img_corners_mu
                img_corners_s2 = np.minimum(img_corners_a, img_corners_b)

                img_corners = np.maximum(img_corners, np.maximum(img_corners_s1, img_corners_s2))

            non_maximum_suppression(img_corners, 1, params.init_loc_thr, r, corners)

    elif detect_method == DetectMethod.HessianResponse:
        # Gaussian smooth
        from scipy.ndimage import gaussian_filter
        gauss_img = gaussian_filter(img, sigma=1.5, truncate=7 / 2)
        hess_img = hessian_response(gauss_img)
        mi_row, mi_col = np.unravel_index(np.argmin(hess_img), hess_img.shape)
        hess_abs = np.abs(hess_img)
        thr = abs(np.min(hess_img)) * params.init_loc_thr

        for r in params.radius:
            non_maximum_suppression(hess_abs, r, thr, r, corners)

    elif detect_method == DetectMethod.LocalizedRadonTransform:
        response_img = _localized_radon_transform(img)
        for r in params.radius:
            non_maximum_suppression(response_img, r, params.init_loc_thr / 10.0, r, corners)

    # Sub-pixel location refinement
    _location_refinement(img, img_du, img_dv, corners)


def convolve2d_border_replicate_fast(img, kernel):
    """FFT-accelerated 2D convolution with replicate-style padding."""
    h, w = img.shape
    kh, kw = kernel.shape

    # Pad image
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    # Flip kernel for convolution (correlation vs convolution)
    flipped = kernel[::-1, ::-1]

    # FFT-based convolution
    result = fftconvolve(padded, flipped, mode='valid')

    # Crop to original size
    ch, cw = result.shape
    if ch > h:
        result = result[:h, :]
    if cw > w:
        result = result[:, :w]

    return result.reshape(h, w)
