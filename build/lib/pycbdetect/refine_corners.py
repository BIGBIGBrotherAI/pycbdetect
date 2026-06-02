"""
Corner refinement: edge orientation estimation + sub-pixel localization.

Estimates dominant edge directions via angular-histogram modes, refines
orientations with eigen-decomposition, then performs Gauss-Newton style
iterative relocation.
"""

import math

import numpy as np

from pycbdetect.config import Corner, CornerType, Params
from pycbdetect.meanshift import find_modes_meanshift
from pycbdetect.utils import get_image_patch, weight_mask


def _edge_orientations(img_angle_sub, img_weight_sub, three=False):
    """Return dominant edge-direction unit vectors from angular histogram.

    Converts normal angles to direction angles (+pi/2), accumulates weighted
    histogram, finds modes, sorts by angle, and returns cosine/sine components.

    Args:
        img_angle_sub: 2D array of gradient angles in patch
        img_weight_sub: 2D array of gradient magnitudes in patch
        three: whether to return 3 directions (MonkeySaddlePoint)

    Returns:
        List of [dx, dy] arrays (length 2 or 3), or empty list on failure.
    """
    n_bins = 32

    # Convert normals → directions
    angles = img_angle_sub.copy()
    angles += math.pi / 2
    angles = np.where(angles >= math.pi, angles - math.pi, angles)

    # Histogram
    angle_hist = np.zeros(n_bins, dtype=np.float64)
    for i in range(angles.shape[1]):
        for j in range(angles.shape[0]):
            bin_idx = int(math.floor(angles[j, i] / (math.pi / n_bins)))
            angle_hist[bin_idx] += img_weight_sub[j, i]

    modes = find_modes_meanshift(angle_hist.tolist(), 1.5)

    if three:
        if len(modes) <= 2:
            return []
        raw_angles = [modes[k][0] * math.pi / n_bins + math.pi / n_bins / 2 for k in range(3)]
    else:
        if len(modes) <= 1:
            return []
        raw_angles = [modes[0][0] * math.pi / n_bins + math.pi / n_bins / 2,
                       modes[1][0] * math.pi / n_bins + math.pi / n_bins / 2]

    # Sort ascending
    raw_angles.sort()

    # Minimum separation check
    deltas = []
    for ai in range(len(raw_angles)):
        for aj in range(ai + 1, len(raw_angles)):
            da = min(abs(raw_angles[aj] - raw_angles[ai]),
                     abs(raw_angles[ai] + math.pi - raw_angles[aj]))
            deltas.append(da)
    thresh = 0.2 if three else 0.3
    if any(d <= thresh for d in deltas):
        return []

    result = [[math.cos(a), math.sin(a)] for a in raw_angles]
    return result


def refine_corners(img_du, img_dv, img_angle, img_weight, corners, params):
    """Iterative sub-pixel refinement of corner positions and orientations.

    Steps:
    1. Estimate edge orientations from angular histogram
    2. Refine orientations via eigen-analysis of gradient covariance
    3. Iteratively relocate corner using constrained Gauss-Newton solve
    """
    max_iter = 5
    eps = 0.01
    is_monkey = params.corner_type == CornerType.MonkeySaddlePoint

    h, w = img_du.shape
    mask_dict = weight_mask(params.radius)

    nc = len(corners.p)
    # Pre-size orientation & score slots so downstream index assignment works
    while len(corners.v1) < nc:
        corners.v1.append(None)
    while len(corners.v2) < nc:
        corners.v2.append(None)
    while len(corners.v3) < nc:
        corners.v3.append(None)
    
    keep = np.zeros(nc, dtype=bool)

    for ci in range(len(corners.p)):
        ui = round(corners.p[ci][0])
        vi = round(corners.p[ci][1])
        u_init = corners.p[ci][0]
        v_init = corners.p[ci][1]
        r = corners.r[ci]

        if ui - r < 0 or ui + r >= w - 1 or vi - r < 0 or vi + r >= h - 1:
            continue

        img_angle_sub = get_image_patch(img_angle, ui, vi, r)
        img_weight_sub = get_image_patch(img_weight, ui, vi, r)
        img_weight_sub *= mask_dict.get(r, np.ones_like(img_weight_sub))

        v_dirs = _edge_orientations(img_angle_sub, img_weight_sub, three=is_monkey)
        if not v_dirs:
            continue

        # Orientation refinement via eigen decomposition
        ndirs = 3 if is_monkey else 2
        As = [np.zeros((2, 2), dtype=np.float64) for _ in range(ndirs)]

        for j2 in range(vi - r, vi + r + 1):
            for i2 in range(ui - r, ui + r + 1):
                o_du = img_du[j2, i2]
                o_dv = img_dv[j2, i2]
                o_norm = math.sqrt(o_du ** 2 + o_dv ** 2)
                if o_norm < 0.1:
                    continue
                od_nrm = o_du / o_norm
                ov_nrm = o_dv / o_norm

                for dd in range(ndirs):
                    proj = abs(od_nrm * v_dirs[dd][0] + ov_nrm * v_dirs[dd][1])
                    if proj < 0.25:
                        As[dd][0, 0] += o_du * o_du
                        As[dd][0, 1] += o_du * o_dv
                        As[dd][1, 0] += o_du * o_dv
                        As[dd][1, 1] += o_dv * o_dv

        for dd in range(ndirs):
            evals, evecs = np.linalg.eigh(As[dd])
            v_dirs[dd] = [evecs[1, 0], evecs[1, 1]]

        # Sort by cross-product sign convention
        for _ in range(len(v_dirs)):
            swapped = False
            for si in range(len(v_dirs) - 1):
                cp = v_dirs[si][0] * v_dirs[si + 1][1] - v_dirs[si][1] * v_dirs[si + 1][0]
                if cp < 0:
                    v_dirs[si], v_dirs[si + 1] = v_dirs[si + 1], v_dirs[si]
                    swapped = True
            if not swapped:
                break

        last_dir = len(v_dirs) - 1
        v_dirs[last_dir] = [-v_dirs[last_dir][0], -v_dirs[last_dir][1]]
        # Re-sort
        for _ in range(len(v_dirs)):
            swapped = False
            for si in range(len(v_dirs) - 1):
                cp = v_dirs[si][0] * v_dirs[si + 1][1] - v_dirs[si][1] * v_dirs[si + 1][0]
                if cp < 0:
                    v_dirs[si], v_dirs[si + 1] = v_dirs[si + 1], v_dirs[si]
                    swapped = True
            if not swapped:
                break

        if params.polynomial_fit:
            keep[ci] = True
            corners.v1[ci] = np.array(v_dirs[0], dtype=np.float64)
            corners.v2[ci] = np.array(v_dirs[1], dtype=np.float64)
            if is_monkey:
                corners.v3[ci] = np.array(v_dirs[2], dtype=np.float64)
            continue

        # Location refinement
        u_cur, v_cur = u_init, v_init
        for nit in range(max_iter):
            G_mat = np.zeros((2, 2), dtype=np.float64)
            b_vec = np.zeros(2, dtype=np.float64)

            if u_cur - r < 0 or u_cur + r >= w or v_cur - r < 0 or v_cur + r >= h:
                break

            img_du_sub = get_image_patch(img_du, u_cur, v_cur, r)
            img_dv_sub = get_image_patch(img_dv, u_cur, v_cur, r)

            for j2 in range(2 * r + 1):
                for i2 in range(2 * r + 1):
                    o_du = img_du_sub[j2, i2]
                    o_dv = img_dv_sub[j2, i2]
                    o_norm = math.sqrt(o_du ** 2 + o_dv ** 2)
                    if o_norm < 0.1:
                        continue
                    od_nrm = o_du / o_norm
                    ov_nrm = o_dv / o_norm
                    if i2 == r and j2 == r:
                        continue

                    accepted = False
                    for dd in range(ndirs):
                        perp_u = (i2 - r) - ((i2 - r) * v_dirs[dd][0] + (j2 - r) * v_dirs[dd][1]) * v_dirs[dd][0]
                        perp_v = (j2 - r) - ((i2 - r) * v_dirs[dd][0] + (j2 - r) * v_dirs[dd][1]) * v_dirs[dd][1]
                        d_perp = math.sqrt(perp_u ** 2 + perp_v ** 2)
                        proj = abs(od_nrm * v_dirs[dd][0] + ov_nrm * v_dirs[dd][1])
                        if d_perp < 3 and proj < 0.25:
                            accepted = True
                            break

                    if accepted:
                        G_mat[0, 0] += o_du * o_du
                        G_mat[0, 1] += o_du * o_dv
                        G_mat[1, 0] += o_du * o_dv
                        G_mat[1, 1] += o_dv * o_dv
                        b_vec[0] += o_du * o_du * (i2 - r + u_cur) + o_du * o_dv * (j2 - r + v_cur)
                        b_vec[1] += o_du * o_dv * (i2 - r + u_cur) + o_dv * o_dv * (j2 - r + v_cur)

            try:
                new_pos = np.linalg.solve(G_mat, b_vec)
            except np.linalg.LinAlgError:
                break

            u_last, v_last = u_cur, v_cur
            u_cur, v_cur = new_pos[0], new_pos[1]
            dist = math.sqrt((u_cur - u_last) ** 2 + (v_cur - v_last) ** 2)
            if dist >= 3:
                u_cur, v_cur = u_last, v_last
                break
            if dist <= eps:
                break

        disp = math.sqrt((u_cur - u_init) ** 2 + (v_cur - v_init) ** 2)
        if disp < max(r / 2, 3):
            keep[ci] = True
            corners.p[ci] = np.array([u_cur, v_cur], dtype=np.float64)
            corners.v1[ci] = np.array(v_dirs[0], dtype=np.float64)
            corners.v2[ci] = np.array(v_dirs[1], dtype=np.float64)
            if is_monkey:
                corners.v3[ci] = np.array(v_dirs[2], dtype=np.float64)

    fp, fr, fv1, fv2, fv3 = [], [], [], [], []
    for ci in range(len(corners.p)):
        if keep[ci]:
            fp.append(corners.p[ci].copy())
            fr.append(corners.r[ci])
            fv1.append(corners.v1[ci].copy())
            fv2.append(corners.v2[ci].copy())
            if is_monkey:
                fv3.append(corners.v3[ci].copy())

    corners.p = fp
    corners.r = fr
    corners.v1 = fv1
    corners.v2 = fv2
    if is_monkey:
        corners.v3 = fv3
