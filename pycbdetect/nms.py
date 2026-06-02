"""
Non-maximum suppression routines.

Two variants: dense grid-based NMS for initial detection, and sparse NMS
based on scores for final pruning.
"""

import math

import numpy as np

from pycbdetect.config import Corner, CornerType, Params


def non_maximum_suppression(img, n, tau, margin, corners):
    """Grid-strided non-maximum suppression on a density map.

    Scans every (n+1)-spaced cell, finds local maxima in an (n+1)x(n+1) tile,
    then verifies against neighborhood of radius n. Keeps peaks exceeding tau.

    Args:
        img: 2D response map (float64)
        n: verification radius
        tau: minimum peak value
        margin: border exclusion
        corners: Corner container to append detections to
    """
    h, w = img.shape
    choose = np.zeros((h, w), dtype=np.uint8)

    step = n + 1
    row_start = step + margin - 1
    nrows = int(math.floor((h - 2 * margin) / step)) + 1

    for ri in range(nrows):
        j_base = ri * step + margin - 1
        for j in range(j_base, h, step):
            for i in range(step + margin, w - n - margin, step):
                # Find max in tile
                maxi, maxj = i, j
                maxval = img[j, i]
                for dj in range(n + 1):
                    for di in range(n + 1):
                        vv = j + dj
                        uu = i + di
                        if 0 <= vv < h and 0 <= uu < w:
                            if img[vv, uu] > maxval:
                                maxi, maxj = uu, vv
                                maxval = img[vv, uu]

                # Verify: nothing bigger in expanded neighborhood
                dominated = False
                for dj in range(-n, n + 1):
                    for di in range(-n, n + 1):
                        vv = maxj + dj
                        uu = maxi + di
                        if 0 <= vv < h - margin and 0 <= uu < w - margin:
                            if img[vv, uu] > maxval:
                                dominated = True
                                break
                    if dominated:
                        break
                if dominated:
                    continue

                if maxval > tau:
                    choose[maxj, maxi] = 1

    for j in range(margin, h - margin):
        for i in range(margin, w - margin):
            if choose[j, i] == 1:
                corners.p.append(np.array([float(i), float(j)], dtype=np.float64))
                corners.r.append(margin)


def non_maximum_suppression_sparse(corners, n, img_shape, params):
    """Score-aware sparse NMS among candidate corners.

    Among overlapping candidates (within nxn window), keep only the highest-scoring.
    """
    h, w = img_shape
    img_score = np.zeros((h, w), dtype=np.float64)
    used = np.full((h, w), -1, dtype=np.int64)

    for i in range(len(corners.p)):
        u = int(round(corners.p[i][0]))
        v = int(round(corners.p[i][1]))
        if 0 <= v < h and 0 <= u < w:
            if img_score[v, u] < corners.score[i]:
                img_score[v, u] = corners.score[i]
                used[v, u] = i

    is_monkey = params.corner_type == CornerType.MonkeySaddlePoint

    out_p, out_r, out_v1, out_v2, out_v3, out_score = [], [], [], [], [], []

    for i in range(len(corners.p)):
        u = int(round(corners.p[i][0]))
        v = int(round(corners.p[i][1]))
        score = corners.score[i]

        if 0 <= v < h and 0 <= u < w and used[v, u] == i:
            dominated = False
            for dj in range(-n, n + 1):
                for di in range(-n, n + 1):
                    vv, uu = v + dj, u + di
                    if 0 <= vv < h and 0 <= uu < w:
                        if img_score[vv, uu] > score and (uu != u or vv != v):
                            dominated = True
                            break
                    if dominated:
                        break
            if dominated:
                continue

            out_p.append(corners.p[i].copy())
            out_r.append(corners.r[i])
            out_v1.append(corners.v1[i].copy())
            out_v2.append(corners.v2[i].copy())
            if is_monkey:
                out_v3.append(corners.v3[i].copy())
            out_score.append(corners.score[i])

    corners.p = out_p
    corners.r = out_r
    corners.v1 = out_v1
    corners.v2 = out_v2
    if is_monkey:
        corners.v3 = out_v3
    corners.score = out_score
