"""
Pre-filter corners based on zero-crossing count and angular histogram modes.

Only corners exhibiting the right number of alternating dark/light sectors
around a circular sampling profile survive.
"""

import math

import numpy as np

from pycbdetect.config import Corner, CornerType, Params
from pycbdetect.meanshift import find_modes_meanshift
from pycbdetect.utils import weight_mask


def filter_corners(img, img_angle, img_weight, corners, params):
    """Remove spurious corners that lack proper sector alternation.

    For each corner, samples intensities on a circle, checks zero-crossing
    count, builds weighted angular histogram, and validates mode count.
    """
    if params.corner_type == CornerType.SaddlePoint:
        n_circle, n_bin = 32, 32
        crossing_thr = 3
        need_crossing = 4
        need_mode = 2
    else:
        n_circle, n_bin = 48, 32
        crossing_thr = 3
        need_crossing = 6
        need_mode = 3

    h, w = img.shape
    cos_v = np.array([math.cos(i * 2.0 * math.pi / (n_circle - 1)) for i in range(n_circle)])
    sin_v = np.array([math.sin(i * 2.0 * math.pi / (n_circle - 1)) for i in range(n_circle)])

    mask_dict = weight_mask(params.radius)

    keep = np.zeros(len(corners.p), dtype=bool)

    for i in range(len(corners.p)):
        center_u = round(corners.p[i][0])
        center_v = round(corners.p[i][1])
        r = corners.r[i]

        if center_u - r < 0 or center_u + r >= w - 1:
            continue
        if center_v - r < 0 or center_v + r >= h - 1:
            continue

        # Sample circle
        c_vals = np.zeros(n_circle, dtype=np.float64)
        for j in range(n_circle):
            circle_u = int(round(center_u + 0.75 * r * cos_v[j]))
            circle_v = int(round(center_v + 0.75 * r * sin_v[j]))
            circle_u = max(0, min(circle_u, w - 1))
            circle_v = max(0, min(circle_v, h - 1))
            c_vals[j] = img[circle_v, circle_u]

        min_c, max_c = c_vals.min(), c_vals.max()
        c_vals = c_vals - min_c - (max_c - min_c) / 2.0

        # Count zero crossings
        num_crossings = 0
        first_cross = 0
        for j in range(n_circle):
            if (c_vals[j] > 0) ^ (c_vals[(j + 1) % n_circle] > 0):
                first_cross = (j + 1) % n_circle
                break

        count_run = 1
        for j in range(first_cross, n_circle + first_cross):
            ji = j % n_circle
            if (c_vals[ji] > 0) ^ (c_vals[(ji + 1) % n_circle] > 0):
                if count_run >= crossing_thr:
                    num_crossings += 1
                count_run = 1
            else:
                count_run += 1

        # Angular histogram
        top_left_u = max(center_u - r, 0)
        top_left_v = max(center_v - r, 0)
        bot_right_u = min(center_u + r, w - 1)
        bot_right_v = min(center_v - r, h - 1)
        bot_right_v = min(center_v + r, h - 1)

        # Extract weight sub-region
        sz = 2 * r + 1
        img_weight_sub = np.zeros((sz, sz), dtype=np.float64)
        dst_j_lo = top_left_v - center_v + r
        dst_j_hi = bot_right_v - center_v + r + 1
        dst_i_lo = top_left_u - center_u + r
        dst_i_hi = bot_right_u - center_u + r + 1
        img_weight_sub[dst_j_lo:dst_j_hi, dst_i_lo:dst_i_hi] = (
            img_weight[top_left_v:bot_right_v + 1, top_left_u:bot_right_u + 1]
        )
        img_weight_sub *= mask_dict.get(r, np.ones_like(img_weight_sub))

        tmp_maxval = img_weight_sub.max()
        img_weight_sub[img_weight_sub < 0.5 * tmp_maxval] = 0.0

        angle_hist = np.zeros(n_bin, dtype=np.float64)
        for j2 in range(top_left_v, bot_right_v + 1):
            for i2 in range(top_left_u, bot_right_u + 1):
                bin_idx = int(math.floor(img_angle[j2, i2] / (math.pi / n_bin))) % n_bin
                angle_hist[bin_idx] += img_weight_sub[j2 - center_v + r, i2 - center_u + r]

        modes = find_modes_meanshift(angle_hist.tolist(), 1.5)
        num_modes = 0
        for _, val in modes:
            if 2 * val > modes[0][1]:
                num_modes += 1

        if num_crossings == need_crossing and num_modes == need_mode:
            keep[i] = True

    filtered_p, filtered_r = [], []
    for i in range(len(corners.p)):
        if keep[i]:
            filtered_p.append(corners.p[i].copy())
            filtered_r.append(corners.r[i])

    corners.p = filtered_p
    corners.r = filtered_r
