"""
Non-maximum suppression routines.

Two variants: dense grid-based NMS for initial detection, and sparse NMS
based on scores for final pruning.
"""

import numpy as np
from scipy.ndimage import maximum_filter

from pycbdetect.config import Corner, CornerType, Params


def non_maximum_suppression(img, n, tau, margin, corners):
    """Grid-strided non-maximum suppression on a density map.

    Preserves original algorithmic semantics while replacing expensive
    per-pixel Python loops with vectorised NumPy/scipy operations wherever
    possible.  Small-tile scans remain looped but operate on tiny arrays.

    Args:
        img: 2D response map (float64)
        n: verification radius
        tau: minimum peak value
        margin: border exclusion
        corners: Corner container to append detections to
    """
    h, w = img.shape
    step = n + 1

    # Stride-grid origins
    row_starts = np.arange(margin - 1, h - n, step)
    col_starts = np.arange(step + margin, w - n - margin, step)

    # Expanded-neighbourhood maximum (used for domination check)
    expand_size = 2 * n + 1
    expand_max = maximum_filter(img, size=expand_size, mode="constant", cval=-np.inf)

    for rs in row_starts:
        for cs in col_starts:
            # Tile bounds
            r_hi = min(rs + step, h)
            c_hi = min(cs + step, w)
            tile = img[rs:r_hi, cs:c_hi]

            # Argmax inside tile
            flat_idx = np.argmax(tile)
            dr, dc = np.divmod(flat_idx, tile.shape[1])
            mr, mc = rs + dr, cs + dc
            mval = tile[dr, dc]

            if mval <= tau:
                continue

            # Domination check: is this truly the max in expanded neighbourhood?
            if expand_max[mr, mc] != mval:
                continue

            # Border check
            if mr < margin or mr >= h - margin or mc < margin or mc >= w - margin:
                continue

            corners.p.append(np.array([float(mc), float(mr)], dtype=np.float64))
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
