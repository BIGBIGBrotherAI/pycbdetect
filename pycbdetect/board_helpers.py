"""
Helper functions for board initialization, growth, energy evaluation, and filtering.

These mirror the C++ implementations in init_board.cc, grow_board.cc,
board_energy.cc, and filter_board.cc.
"""

import math
import random
import time

import numpy as np

from pycbdetect.config import Board, Corner, CornerType, GrowType, Params


# ---------------------------------------------------------------------------
# Directional neighbour selection
# ---------------------------------------------------------------------------

def _directional_neighbor(corners, used, idx, v, min_dist_ref):
    """Find unused corner closest to ray *v* emanating from corner *idx*.

    Distance metric penalises lateral deviation from the ray heavily.
    """
    dists = np.full(len(corners.p), 1e10, dtype=np.float64)
    pc = corners.p[idx]

    for i in range(len(corners.p)):
        if used[i]:
            continue
        dir_vec = corners.p[i] - pc
        dist_along = float(np.dot(dir_vec, v))
        lat_vec = dir_vec - dist_along * v
        dist_lat = float(np.sqrt(lat_vec.dot(lat_vec)))
        if dist_along >= 0:
            dists[i] = dist_along + 5.0 * dist_lat

    nb = int(np.argmin(dists))
    min_dist_ref[0] = dists[nb]
    return nb


# ---------------------------------------------------------------------------
# Board initialisation (3×3 seed)
# ---------------------------------------------------------------------------

def init_board(corners, used, board, idx):
    """Try to initialise a 3×3 board centred on corner *idx*.

    Marks neighbours in *used*, populates board.idx and board.energy.
    Returns True on success.
    """
    board.idx = []
    board.energy = []

    if len(corners.p) < 9:
        return False

    v1 = corners.v1[idx]
    v2 = corners.v3[idx] if (corners.v3 and corners.v3[idx] is not None) else corners.v2[idx]
    if v1 is None or v2 is None:
        return False

    board.idx = [[0]*3 for _ in range(3)]
    board.idx[1][1] = idx
    used[idx] = 1

    md = [0.0] * 8

    # Left / Right / Top / Bottom
    board.idx[1][0] = _directional_neighbor(corners, used, idx, -v1, md[0:1])
    used[board.idx[1][0]] = 1
    board.idx[1][2] = _directional_neighbor(corners, used, idx, v1, md[1:2])
    used[board.idx[1][2]] = 1
    board.idx[0][1] = _directional_neighbor(corners, used, idx, -v2, md[2:3])
    used[board.idx[0][1]] = 1
    board.idx[2][1] = _directional_neighbor(corners, used, idx, v2, md[3:4])
    used[board.idx[2][1]] = 1

    # Diagonals – resolve conflicts by choosing equidistant candidate
    diag_pairs = [
        (board.idx[1][0], -v2, board.idx[0][1], -v1, 0, 0, 4),
        (board.idx[1][2], -v2, board.idx[0][1], v1, 0, 2, 5),
        (board.idx[1][0], v2, board.idx[2][1], -v1, 2, 0, 6),
        (board.idx[1][2], v2, board.idx[2][1], v1, 2, 2, 7),
    ]
    for ci_src, vd1, cj_src, vd2, rr, cc, mdi in diag_pairs:
        tmp1 = _directional_neighbor(corners, used, ci_src, vd1, [0.0])
        tmp2 = _directional_neighbor(corners, used, cj_src, vd2, [0.0])
        if tmp1 != tmp2:
            d1 = abs(float(np.linalg.norm(corners.p[tmp1] - corners.p[ci_src]))
                     - float(np.linalg.norm(corners.p[tmp1] - corners.p[cj_src])))
            d2 = abs(float(np.linalg.norm(corners.p[tmp2] - corners.p[ci_src]))
                     - float(np.linalg.norm(corners.p[tmp2] - corners.p[cj_src])))
            if d1 > d2:
                tmp1, tmp2 = tmp2, tmp1
        board.idx[rr][cc] = tmp1
        used[tmp1] = 1

    # Homogeneity check
    for d in md:
        if abs(d - 1e10) < 1:
            for jj in range(3):
                for ii in range(3):
                    used[board.idx[jj][ii]] = 0
            board.idx = []
            return False

    board.num = 9
    board.energy = [[[float('inf')] * 3 for _ in range(3)] for _ in range(3)]
    return True


# ---------------------------------------------------------------------------
# Predict corners from trajectory
# ---------------------------------------------------------------------------

def _predict_corners(corners, p1, p2, p3):
    """Predict next corner positions using replica extrapolation."""
    preds = []
    for k in range(len(p3)):
        if not p1:
            pred = 2 * corners.p[p3[k]] - corners.p[p2[k]]
        else:
            v1 = corners.p[p2[k]] - corners.p[p1[k]]
            v2 = corners.p[p3[k]] - corners.p[p2[k]]
            a1 = math.atan2(v1[1], v1[0])
            a2 = math.atan2(v2[1], v2[0])
            a3 = 2 * a2 - a1
            s1 = float(np.linalg.norm(v1))
            s2 = float(np.linalg.norm(v2))
            s3 = 2 * s2 - s1
            cx = corners.p[p3[k]][0] + 0.75 * s3 * math.cos(a3)
            cy = corners.p[p3[k]][1] + 0.75 * s3 * math.sin(a3)
            pred = np.array([cx, cy], dtype=np.float64)
        preds.append(pred)
    return preds


def _predict_board_corners(corners, used, p1, p2, p3):
    """Associate predicted positions with real (unused) corners greedily."""
    pred_pts = _predict_corners(corners, p1, p2, p3)
    pred_idx = [-2] * len(pred_pts)

    # Distance matrix
    D = np.full((len(pred_pts), len(corners.p)), float('inf'), dtype=np.float64)
    for i in range(len(pred_pts)):
        w = pred_pts[i] - corners.p[p3[i]]
        ww = float(np.linalg.norm(w))
        if ww < 1e-10:
            continue
        wn = w / ww
        wp = np.array([-wn[1], wn[0]])
        for j in range(len(corners.p)):
            if used[j]:
                continue
            vt = corners.p[j] - corners.p[p3[i]]
            va = np.array([float(np.dot(vt, wn)), float(np.dot(vt, wp))]) / (ww * ww)
            d1 = abs(math.atan2(va[1], va[0]))
            d2 = abs(1 - va[0])
            D[i, j] = math.sqrt(d1 + d2 ** 4)

    assigned_rows = set()
    assigned_cols = set()
    while True:
        best_d = float('inf')
        best_rc = None
        for i in range(len(pred_pts)):
            avail = [j for j in range(len(corners.p)) if j not in assigned_cols]
            if not avail:
                continue
            mi = min(avail, key=lambda j: D[i, j])
            if D[i, mi] < best_d:
                best_d = D[i, mi]
                best_rc = (i, mi)
        if best_rc is None or best_d >= float('inf'):
            break
        ir, ic = best_rc
        pred_idx[ir] = ic
        used[ic] = 1
        assigned_rows.add(ir)
        assigned_cols.add(ic)

    return pred_idx


# ---------------------------------------------------------------------------
# Board energy
# ---------------------------------------------------------------------------

def board_energy(corners, board, params):
    """Evaluate structural energy of a board hypothesis.

    Energy = −N × (1 − E_struct) evaluated along three directions.
    Returns (rx, ry, rz) locating the worst triple.
    """
    E_corners = -1.0 * board.num
    max_ES = float('-inf')
    rx, ry, rz = 0, 0, 0

    rows = len(board.idx)
    cols = len(board.idx[0]) if rows > 0 else 0

    # Walk v1 (horizontal triples)
    for i in range(rows):
        for j in range(cols - 2):
            id1, id2, id3 = board.idx[i][j], board.idx[i][j+1], board.idx[i][j+2]
            if id1 >= 0 and id2 >= 0 and id3 >= 0:
                ES = float(np.linalg.norm(corners.p[id1] + corners.p[id3] - 2*corners.p[id2])) / max(
                    float(np.linalg.norm(corners.p[id1] - corners.p[id3])), 1e-10)
                board.energy[i][j][0] = E_corners * (1 - ES)
                if ES > max_ES:
                    max_ES, rx, ry, rz = ES, j, i, 0

    # Walk diagonal (only for monkey saddle)
    if params.corner_type == CornerType.MonkeySaddlePoint:
        for i in range(rows - 2):
            for j in range(cols - 2):
                id1, id2, id3 = board.idx[i][j], board.idx[i+1][j+1], board.idx[i+2][j+2]
                if id1 >= 0 and id2 >= 0 and id3 >= 0:
                    ES = float(np.linalg.norm(corners.p[id1] + corners.p[id3] - 2*corners.p[id2])) / max(
                        float(np.linalg.norm(corners.p[id1] - corners.p[id3])), 1e-10)
                    board.energy[i][j][1] = E_corners * (1 - ES)
                    if ES > max_ES:
                        max_ES, rx, ry, rz = ES, j, i, 1

    # Walk v3 (vertical triples)
    for i in range(rows - 2):
        for j in range(cols):
            id1, id2, id3 = board.idx[i][j], board.idx[i+1][j], board.idx[i+2][j]
            if id1 >= 0 and id2 >= 0 and id3 >= 0:
                ES = float(np.linalg.norm(corners.p[id1] + corners.p[id3] - 2*corners.p[id2])) / max(
                    float(np.linalg.norm(corners.p[id1] - corners.p[id3])), 1e-10)
                board.energy[i][j][2] = E_corners * (1 - ES)
                if ES > max_ES:
                    max_ES, rx, ry, rz = ES, j, i, 2

    return rx, ry, rz


# ---------------------------------------------------------------------------
# Find minimal energy in neighbourhood
# ---------------------------------------------------------------------------

def _find_minE(board, px, py):
    es = board.energy[py][px]
    minE = min(es[0], es[1], es[2])
    # Check neighbouring energies in upstream directions
    if px - 1 >= 0:
        minE = min(minE, board.energy[py][px - 1][0])
    if px - 1 >= 0 and py - 1 >= 0:
        minE = min(minE, board.energy[py - 1][px - 1][1])
    if py - 1 >= 0:
        minE = min(minE, board.energy[py - 1][px][2])
    if px - 2 >= 0:
        minE = min(minE, board.energy[py][px - 2][0])
    if px - 2 >= 0 and py - 2 >= 0:
        minE = min(minE, board.energy[py - 2][px - 2][1])
    if py - 2 >= 0:
        minE = min(minE, board.energy[py - 2][px][2])
    return minE


# ---------------------------------------------------------------------------
# Filter board proposals
# ---------------------------------------------------------------------------

def filter_board(corners, used, board, proposal, energy, params):
    """Prune bad corners from newly proposed additions."""
    while proposal:
        rx, ry, rz = board_energy(corners, board, params)
        p_e = board.energy[ry][rx][rz]
        if p_e <= energy:
            energy[0] = p_e
            break
        if params.corner_type == CornerType.SaddlePoint and not params.occlusion:
            for p in proposal:
                used[board.idx[p[1]][p[0]]] = 0
                board.idx[p[1]][p[0]] = -2
                board.num -= 1
            return

        # Identify worst corner among the three-triple members
        ps = [(rx, ry)]
        if rz == 0:
            ps.append((rx+1, ry)); ps.append((rx+2, ry))
        elif rz == 1:
            ps.append((rx+1, ry+1)); ps.append((rx+2, ry+2))
        else:
            ps.append((rx, ry+1)); ps.append((rx, ry+2))

        minEs = [_find_minE(board, px, py) for px, py in ps]

        best_me = float('-inf')
        best_iter = -1
        for ip, p in enumerate(proposal):
            px, py = p[0], p[1]
            for qi, qps in enumerate(ps):
                if px == qps[0] and py == qps[1] and minEs[qi] > best_me:
                    best_me = minEs[qi]
                    best_iter = ip

        if best_iter >= 0:
            rp = proposal.pop(best_iter)
            used[board.idx[rp[1]][rp[0]]] = 0
            board.idx[rp[1]][rp[0]] = -2
            board.num -= 1


# ---------------------------------------------------------------------------
# Add boundary row/column
# ---------------------------------------------------------------------------

def _add_board_boundary(board, direction):
    """Extend board.grid by one row/column on the requested side."""
    rows = len(board.idx)
    cols = len(board.idx[0]) if rows > 0 else 0
    added = False

    if direction == 0:  # top
        for i in range(cols):
            if board.idx[0][i] not in (-2, -1):
                added = True; break
        if added:
            board.idx.insert(0, [-1]*cols)
            board.energy.insert(0, [[float('inf')]*3 for _ in range(cols)])

    elif direction == 1:  # left
        for i in range(rows):
            if board.idx[i][0] not in (-2, -1):
                added = True; break
        if added:
            for i in range(rows):
                board.idx[i].insert(0, -1)
                board.energy[i].insert(0, [float('inf')]*3)

    elif direction == 2:  # bottom
        for i in range(cols):
            if board.idx[rows-1][i] not in (-2, -1):
                added = True; break
        if added:
            board.idx.append([-1]*cols)
            board.energy.append([[float('inf')]*3 for _ in range(cols)])

    elif direction == 3:  # right
        for i in range(rows):
            if board.idx[i][cols-1] not in (-2, -1):
                added = True; break
        if added:
            for i in range(rows):
                board.idx[i].append(-1)
                board.energy[i].append([float('inf')]*3)

    return added


# ---------------------------------------------------------------------------
# Grow board
# ---------------------------------------------------------------------------

def grow_board(corners, used, board, direction, params):
    """Attempt to extend the board in *direction*.

    Directions 0-3: top/left/bottom/right boundaries
    Directions 4-5: diagonals (monkey saddle only)

    Returns (GrowType, proposal_list_of_xy_tuples).
    """
    if not board.idx:
        return GrowType.Failure, []

    rows = len(board.idx)
    cols = len(board.idx[0])
    proposal = []
    p1, p2, p3 = [], [], []

    # ---- Fill interior holes ------------------------------------------
    if params.corner_type == CornerType.MonkeySaddlePoint or params.occlusion:
        filled_interior = _try_fill_interior(board, corners, used, direction,
                                             proposal, p1, p2, p3, params)
        if filled_interior:
            pred = _predict_board_corners(corners, used, p1, p2, p3)
            board.num += len(proposal)
            for ip, pp in enumerate(proposal):
                if pred[ip] == -2:
                    board.num -= 1
                board.idx[pp[1]][pp[0]] = pred[ip]
            return GrowType.Inside, proposal

    # ---- Extend boundary ----------------------------------------------
    if not _add_board_boundary(board, direction):
        return GrowType.Failure, []

    p1, p2, p3 = [], [], []
    proposal = []
    rows = len(board.idx)
    cols = len(board.idx[0])

    if direction == 0:
        for i in range(cols):
            a, b, c = board.idx[3][i], board.idx[2][i], board.idx[1][i]
            if a < 0 or b < 0 or c < 0:
                continue
            p1.append(a); p2.append(b); p3.append(c)
            proposal.append((i, 0))
    elif direction == 1:
        for i in range(rows):
            a, b, c = board.idx[i][3], board.idx[i][2], board.idx[i][1]
            if a < 0 or b < 0 or c < 0:
                continue
            p1.append(a); p2.append(b); p3.append(c)
            proposal.append((0, i))
    elif direction == 2:
        for i in range(cols):
            a, b, c = board.idx[rows-4][i], board.idx[rows-3][i], board.idx[rows-2][i]
            if a < 0 or b < 0 or c < 0:
                continue
            p1.append(a); p2.append(b); p3.append(c)
            proposal.append((i, rows-1))
    elif direction == 3:
        for i in range(rows):
            a, b, c = board.idx[i][cols-4], board.idx[i][cols-3], board.idx[i][cols-2]
            if a < 0 or b < 0 or c < 0:
                continue
            p1.append(a); p2.append(b); p3.append(c)
            proposal.append((cols-1, i))

    if not proposal:
        return GrowType.Failure, []

    pred = _predict_board_corners(corners, used, p1, p2, p3)

    if params.corner_type == CornerType.SaddlePoint and not params.occlusion:
        fail_flag = False
        for ip in range(len(proposal)):
            if pred[ip] == -2:
                fail_flag = True
                break
        if fail_flag:
            board.num -= len(proposal)
            return GrowType.Failure, []

    board.num += len(proposal)
    for ip, pp in enumerate(proposal):
        if pred[ip] == -2:
            board.num -= 1
        board.idx[pp[1]][pp[0]] = pred[ip]

    return GrowType.Boundary, proposal


def _try_fill_interior(board, corners, used, direction,
                       proposal, p1, p2, p3, params):
    """Fill interior gaps in four cardinal / two diagonal directions.

    Mutates proposal, p1, p2, p3 lists in-place.
    Returns True if any gap was found.
    """
    rows = len(board.idx)
    cols = len(board.idx[0])

    # Helper closures for scanning
    scans = []

    if direction == 0:  # top → downward
        scans.append(lambda: _scan_axis(board, rows, cols, lambda i,j: (i+3,i+2,i+1,j), -1, 0, proposal, p1, p2, p3))
        scans.append(lambda: _scan_axis(board, rows, cols, lambda i,j: (None,i+2,i+1,j), -1, 0, proposal, p2, p3, None))
    elif direction == 1:  # left → rightward
        scans.append(lambda: _scan_axis(board, rows, cols, lambda i,j: (i,j+3,j+2,j+1), 0, -1, proposal, p1, p2, p3))
        scans.append(lambda: _scan_axis(board, rows, cols, lambda i,j: (i,None,j+2,j+1), 0, -1, proposal, p2, p3, None))
    elif direction == 2:  # bottom → upward
        scans.append(lambda: _scan_axis(board, rows, cols, lambda i,j: (i-3,i-2,i-1,j), 1, 0, proposal, p1, p2, p3))
        scans.append(lambda: _scan_axis(board, rows, cols, lambda i,j: (None,i-2,i-1,j), 1, 0, proposal, p2, p3, None))
    elif direction == 3:  # right → leftward
        scans.append(lambda: _scan_axis(board, rows, cols, lambda i,j: (i,j-3,j-2,j-1), 0, 1, proposal, p1, p2, p3))
        scans.append(lambda: _scan_axis(board, rows, cols, lambda i,j: (i,None,j-2,j-1), 0, 1, proposal, p2, p3, None))
    elif direction == 4:  # TL→BR diagonal
        scans.append(lambda: _scan_diag(board, rows, cols, -1, proposal, p1, p2, p3))
        scans.append(lambda: _scan_diag_short(board, rows, cols, -1, proposal, p2, p3))
    elif direction == 5:  # BR→TL diagonal
        scans.append(lambda: _scan_diag(board, rows, cols, 1, proposal, p1, p2, p3))
        scans.append(lambda: _scan_diag_short(board, rows, cols, 1, proposal, p2, p3))

    for fn in scans:
        found = fn()
        if found or params.strict_grow:
            return found

    return False


def _scan_axis(board, rows, cols, tri_fn, sweep_first, sweep_second,
               proposal, pa, pb, pc):
    """Generic axial scanner. Returns True if any slot placed."""
    rows = len(board.idx)
    cols = len(board.idx[0])
    found = False
    # Three-anchor pass
    if sweep_first == -1:
        i_range = range(rows-4, -1, -1)
    elif sweep_first == 1:
        i_range = range(3, rows)
    else:
        i_range = range(rows)

    if sweep_second == -1:
        j_range = range(cols-4, -1, -1)
    elif sweep_second == 1:
        j_range = range(3, cols)
    else:
        j_range = range(cols)

    for i in i_range:
        for j in j_range:
            if board.idx[i][j] != -1:
                continue
            ids = tri_fn(i, j)
            if pc is not None:
                ia, ib_, ic = ids[0], ids[1], ids[2]
                jc = ids[3]
                if ia < 0 or ib_ < 0 or ic < 0:
                    continue
                pa.append(board.idx[ia][jc]); pb.append(board.idx[ib_][jc]); pc.append(board.idx[ic][jc])
            else:
                ib_, ic = ids[1], ids[2]
                jc = ids[3]
                if ib_ < 0 or ic < 0:
                    continue
                pb.append(board.idx[ib_][jc]); pc.append(board.idx[ic][jc])
            proposal.append((j, i))
            found = True
    return found


def _scan_diag(board, rows, cols, sign, proposal, p1, p2, p3):
    found = False
    if sign == -1:
        for i in range(3, rows):
            for j in range(3, cols):
                if board.idx[i][j] != -1:
                    continue
                a = board.idx[i-3][j-3]
                b = board.idx[i-2][j-2]
                c = board.idx[i-1][j-1]
                if a < 0 or b < 0 or c < 0:
                    continue
                p1.append(a); p2.append(b); p3.append(c)
                proposal.append((j, i))
                found = True
    else:
        for i in range(rows-3):
            for j in range(cols-3):
                if board.idx[i][j] != -1:
                    continue
                a = board.idx[i+3][j+3]
                b = board.idx[i+2][j+2]
                c = board.idx[i+1][j+1]
                if a < 0 or b < 0 or c < 0:
                    continue
                p1.append(a); p2.append(b); p3.append(c)
                proposal.append((j, i))
                found = True
    return found


def _scan_diag_short(board, rows, cols, sign, proposal, p2, p3):
    found = False
    if sign == -1:
        for i in range(2, rows):
            for j in range(2, cols):
                if board.idx[i][j] != -1:
                    continue
                b = board.idx[i-2][j-2]
                c = board.idx[i-1][j-1]
                if b < 0 or c < 0:
                    continue
                p2.append(b); p3.append(c)
                proposal.append((j, i))
                found = True
    else:
        for i in range(rows-2):
            for j in range(cols-2):
                if board.idx[i][j] != -1:
                    continue
                b = board.idx[i+2][j+2]
                c = board.idx[i+1][j+1]
                if b < 0 or c < 0:
                    continue
                p2.append(b); p3.append(c)
                proposal.append((j, i))
                found = True
    return found
