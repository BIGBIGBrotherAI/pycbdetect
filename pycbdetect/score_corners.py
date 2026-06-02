"""
Corner scoring and removal of low-quality detections.

Scores each corner by combining gradient alignment and template-intensity
matching metrics. Then removes corners below threshold.
"""

import math

import numpy as np

from pycbdetect.config import Corner, CornerType, Params
from pycbdetect.utils import (
    create_correlation_patch_2angle,
    create_correlation_patch_3angle,
    get_image_patch,
    weight_mask,
)


def _corner_correlation_score_2dir(img, img_weight, v1, v2):
    """Correlation score for saddle-point corners (2 edge dirs)."""
    center = (img.shape[1] - 1) / 2.0

    # Gradient filter kernel
    img_filter = -np.ones_like(img)
    for u in range(img.shape[1]):
        for v in range(img.shape[0]):
            p1 = np.array([u - center, v - center])
            p2 = float(np.dot(p1, v1)) * v1
            p3 = float(np.dot(p1, v2)) * v2
            d2 = math.sqrt(((p1 - p2) ** 2).sum())
            d3 = math.sqrt(((p1 - p3) ** 2).sum())
            if d2 <= 1.5 or d3 <= 1.5:
                img_filter[v, u] = 1.0

    # Normalize
    mf, sf = img_filter.mean(), img_filter.std()
    if sf > 1e-10:
        img_filter = (img_filter - mf) / sf
    mw, sw = img_weight.mean(), img_weight.std()
    if sw > 1e-10:
        iw_norm = (img_weight - mw) / sw
    else:
        iw_norm = img_weight - mw

    score_grad = float(np.sum(iw_norm * img_filter))
    score_grad = max(score_grad / (img.size - 1), 0.0)

    # Intensity filter
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    rk = (img.shape[1] - 1) // 2
    kernels = create_correlation_patch_2angle(a1, a2, rk)

    a1v = float(np.sum(img * kernels[0]))
    a2v = float(np.sum(img * kernels[1]))
    b1v = float(np.sum(img * kernels[2]))
    b2v = float(np.sum(img * kernels[3]))

    mu = (a1v + a2v + b1v + b2v) / 4.0

    s1 = min(min(a1v, a2v) - mu, mu - max(b1v, b2v))
    s2 = min(mu - max(a1v, a2v), min(b1v, b2v) - mu)
    score_intens = max(max(s1, s2), 0.0)

    return score_grad * score_intens


def _corner_correlation_score_3dir(img, img_weight, v1, v2, v3):
    """Correlation score for monkey-saddle corners (3 edge dirs)."""
    center = (img.shape[1] - 1) / 2.0

    img_filter = -np.ones_like(img)
    for u in range(img.shape[1]):
        for v in range(img.shape[0]):
            p1 = np.array([u - center, v - center])
            p2 = float(np.dot(p1, v1)) * v1
            p3 = float(np.dot(p1, v2)) * v2
            p4 = float(np.dot(p1, v3)) * v3
            d2 = math.sqrt(((p1 - p2) ** 2).sum())
            d3 = math.sqrt(((p1 - p3) ** 2).sum())
            d4 = math.sqrt(((p1 - p4) ** 2).sum())
            if d2 <= 1.5 or d3 <= 1.5 or d4 <= 1.5:
                img_filter[v, u] = 1.0

    mf, sf = img_filter.mean(), img_filter.std()
    if sf > 1e-10:
        img_filter = (img_filter - mf) / sf
    mw, sw = img_weight.mean(), img_weight.std()
    if sw > 1e-10:
        iw_norm = (img_weight - mw) / sw
    else:
        iw_norm = img_weight - mw

    score_grad = float(np.sum(iw_norm * img_filter))
    score_grad = max(score_grad / (img.size - 1), 0.0)

    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    a3 = math.atan2(v3[1], v3[0])
    rk = (img.shape[1] - 1) // 2
    kernels = create_correlation_patch_3angle(a1, a2, a3, rk)

    vals = [float(np.sum(img * kernels[k])) for k in range(6)]
    mu = sum(vals) / 6.0
    min_a = min(vals[0], vals[1], vals[2])
    min_b = min(vals[3], vals[4], vals[5])

    s1 = min(min_a - mu, mu - min_b)
    s2 = min(mu - min_a, min_b - mu)
    score_intens = max(max(s1, s2), 0.0)

    return score_grad * score_intens


def score_corners(img, img_weight, corners, params):
    """Assign quality scores to each corner."""
    corners.score = [0.0] * len(corners.p)
    h, w = img.shape
    mask_dict = weight_mask(params.radius)

    for i in range(len(corners.p)):
        u = corners.p[i][0]
        v = corners.p[i][1]
        r = corners.r[i]

        if u - r < 0 or u + r >= w - 1 or v - r < 0 or v + r >= h - 1:
            corners.score[i] = 0.0
            continue

        img_sub = get_image_patch(img, u, v, r)
        iw_sub = get_image_patch(img_weight, u, v, r)
        iw_sub *= mask_dict.get(r, np.ones_like(iw_sub))

        if params.corner_type == CornerType.SaddlePoint:
            corners.score[i] = _corner_correlation_score_2dir(
                img_sub, iw_sub, corners.v1[i], corners.v2[i]
            )
        else:
            corners.score[i] = _corner_correlation_score_3dir(
                img_sub, iw_sub, corners.v1[i], corners.v2[i], corners.v3[i]
            )


def remove_low_scoring_corners(threshold, corners, params):
    """Discard corners whose score falls below threshold."""
    is_monkey = params.corner_type == CornerType.MonkeySaddlePoint
    fp, fr, fv1, fv2, fv3, fs = [], [], [], [], [], []
    for i in range(len(corners.p)):
        if corners.score[i] > threshold:
            fp.append(corners.p[i].copy())
            fr.append(corners.r[i])
            fv1.append(corners.v1[i].copy())
            fv2.append(corners.v2[i].copy())
            if is_monkey:
                fv3.append(corners.v3[i].copy())
            fs.append(corners.score[i])

    corners.p = fp
    corners.r = fr
    corners.v1 = fv1
    corners.v2 = fv2
    if is_monkey:
        corners.v3 = fv3
    corners.score = fs
