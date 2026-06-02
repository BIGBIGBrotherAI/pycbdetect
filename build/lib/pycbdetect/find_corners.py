"""
Top-level corner detection pipeline.

Orchestrates: normalization → initial detection → filtering → refinement →
polynomial fit → scoring → NMS.
Also runs a scaled-image pass for complementary detections.
"""

import sys

import numpy as np

from pycbdetect.config import Corner, Params
from pycbdetect.filter_corners import filter_corners
from pycbdetect.get_init_location import get_init_location
from pycbdetect.imgproc import image_normalization_and_gradients
from pycbdetect.nms import non_maximum_suppression_sparse
from pycbdetect.polynomial_fit import polynomial_fit
from pycbdetect.refine_corners import refine_corners
from pycbdetect.score_corners import remove_low_scoring_corners, score_corners


def _grayscale_double(img):
    """Convert image to float64 grayscale in [0,1]."""
    if len(img.shape) == 3 and img.shape[2] >= 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img[..., 0] if len(img.shape) == 3 else img
    return gray.astype(np.float64) / 255.0


def _resize_nearest_scale(img, scale):
    """Resize image by integer/fractional scale using nearest interpolation."""
    h, w = img.shape[:2]
    nh, nw = int(round(h * scale)), int(round(w * scale))
    nc = img.shape[2] if len(img.shape) == 3 else 1
    if nc > 1:
        result = np.zeros((nh, nw, nc), dtype=img.dtype)
    else:
        result = np.zeros((nh, nw), dtype=img.dtype)
    for axis_dim, (out_sz, in_sz) in enumerate(zip(result.shape[:2], img.shape[:2])):
        ratio = in_sz / out_sz
    # Simple approach: iterate output coords
    for oy in range(nh):
        for ox in range(nw):
            sy = min(int(oy * h / nh), h - 1)
            sx = min(int(ox * w / nw), w - 1)
            if nc > 1:
                result[oy, ox] = img[sy, sx]
            else:
                result[oy, ox] = img[sy, sx]
    return result


def _run_pipeline_on_scaled(img_orig_gray, scale, params):
    """Run full detection pipeline on a possibly-rescaled image.

    Returns corners expressed in ORIGINAL image coordinates.
    """
    if scale != 1.0:
        img_scaled = _resize_nearest_scale(img_orig_gray, scale)
    else:
        img_scaled = img_orig_gray.copy()

    img_norm, img_du, img_dv, img_angle, img_weight = (
        image_normalization_and_gradients(img_scaled, params)
    )

    corners = Corner()

    # Get initial locations
    get_init_location(img_norm, img_du, img_dv, corners, params)
    if len(corners.p) == 0:
        return corners

    if params.show_processing:
        print(f"  Init ({img_norm.shape[1]}×{img_norm.shape[0]}) … {len(corners.p)} corners", file=sys.stderr)

    # Pre-filter
    filter_corners(img_norm, img_angle, img_weight, corners, params)
    if params.show_processing:
        print(f"  Filter … {len(corners.p)} corners", file=sys.stderr)

    # Refine
    refine_corners(img_du, img_dv, img_angle, img_weight, corners, params)
    if params.show_processing:
        print(f"  Refine … {len(corners.p)} corners", file=sys.stderr)

    # Polynomial fit
    if params.polynomial_fit:
        polynomial_fit(img_norm, corners, params)
        if params.show_processing:
            print(f"  PolyFit … {len(corners.p)} corners", file=sys.stderr)

    # Score
    score_corners(img_norm, img_weight, corners, params)
    remove_low_scoring_corners(params.score_thr, corners, params)

    # Sparse NMS
    non_maximum_suppression_sparse(corners, 3, img_norm.shape, params)

    # Rescale corner positions back to original image space
    if scale != 1.0:
        for p in corners.p:
            p[:] /= scale

    return corners


def find_corners(img, corners=None, params=None):
    """Detect checkerboard/deltille corners in an image.

    Pipeline summary:
    1. Convert to grayscale [0,1]
    2. Normalise + compute gradients
    3. Initialise corner hypotheses (template/Hessian/Radon)
    4. Filter by zero-crossing + angular-mode criteria
    5. Refine positions and orientations iteratively
    6. Quadratic/cubic polynomial surface fit
    7. Correlation-score each corner, discard weak ones
    8. Non-maximum-suppress remaining candidates
    9. Repeat on upscaled/downscaled image and merge

    Args:
        img: Input image (uint8, RGB/BGR or grayscale). Accepts PIL Image or
             numpy ndarray.
        corners: Optional pre-existing Corner container (will be cleared).
        params: Optional Params instance; defaults applied internally.

    Returns:
        Corner container populated with detected corners.
    """
    if isinstance(img, np.ndarray):
        pass
    else:
        # Assume PIL.Image
        img = np.array(img)

    if params is None:
        params = Params()
    if corners is None:
        corners = Corner()
    else:
        corners.clear()

    img_gray = _grayscale_double(img)

    # Primary pass at native resolution
    primary = _run_pipeline_on_scaled(img_gray, 1.0, params)
    for attr in ['p', 'r', 'v1', 'v2', 'v3', 'score']:
        getattr(corners, attr).extend(getattr(primary, attr))

    if params.show_processing:
        print(f"  Native … {len(corners.p)} corners", file=sys.stderr)

    # Secondary pass at alternate scale
    if img_gray.shape[0] < 640 and img_gray.shape[1] < 480:
        sc = 2.0
    elif img_gray.shape[0] >= 640 or img_gray.shape[1] >= 480:
        sc = 0.5
    else:
        sc = None

    if sc is not None:
        secondary = _run_pipeline_on_scaled(img_gray, sc, params)
        min_dist_thr = 3.0 if sc > 1 else 5.0
        for si in range(len(secondary.p)):
            sp = secondary.p[si]
            min_d = float('inf')
            for cp in corners.p:
                d = float(np.sqrt(((sp - cp) ** 2).sum()))
                if d < min_d:
                    min_d = d
            if min_d > min_dist_thr:
                corners.p.append(sp.copy())
                corners.r.append(secondary.r[si])
                corners.v1.append(secondary.v1[si].copy())
                corners.v2.append(secondary.v2[si].copy())
                if hasattr(secondary, 'v3') and secondary.v3 and secondary.v3[si] is not None:
                    corners.v3.append(secondary.v3[si].copy())

        if params.show_processing:
            print(f"  Merge … {len(corners.p)} corners", file=sys.stderr)

    # Final scoring/NMS on merged set
    img_norm, img_du, img_dv, img_angle, img_weight = (
        image_normalization_and_gradients(img_gray.copy(), params)
    )
    score_corners(img_norm, img_weight, corners, params)
    remove_low_scoring_corners(params.score_thr, corners, params)
    non_maximum_suppression_sparse(corners, 3, img_norm.shape, params)

    if params.show_processing:
        print(f"  Final … {len(corners.p)} corners", file=sys.stderr)

    return corners
