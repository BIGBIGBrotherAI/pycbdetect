"""
Boards-from-corners pipeline.

Given a set of scored corners, assemble them into rectangular checkerboard
(or deltilte) hypotheses by seeding 3×3 grids, growing outward, and
evaluating structural energy.
"""

import sys

import numpy as np

from pycbdetect.board_helpers import (
    board_energy,
    filter_board,
    grow_board,
    init_board,
)
from pycbdetect.config import Board, Corner, CornerType, Params


def boards_from_corners(img, corners, boards=None, params=None):
    """Group detected corners into coherent checkerboard patterns.

    Algorithm outline:
    1. Pick a random seed corner
    2. Try to form a 3×3 board around it
    3. Evaluate energy – reject if too high (> -6)
    4. Greedily grow in 4 (or 6) directions
    5. Filter bad additions
    6. Handle overlap with previously found boards

    Args:
        img: Original colour/grayscale image (used only for debug visuals).
        corners: Populated Corner container.
        boards: Output list of Board objects (cleared on entry).
        params: Parameters controlling behaviour.

    Returns:
        List of Board instances.
    """
    if params is None:
        params = Params()
    if boards is None:
        boards = []

    boards.clear()
    used = [0] * len(corners.p)

    rng = np.random.RandomState(42)
    start = rng.randint(0, max(len(corners.p), 1))

    n = 0
    while n < len(corners.p):
        n += 1
        i = (n + start) % len(corners.p)
        if used[i] or not init_board(corners, used, Board(), i):
            continue

        board = locals()['board']  # noqa – shadow avoided
        # Actually grab the board variable correctly
        # (re-init since init_board mutates passed-in board)
        board = Board()
        if not init_board(corners, used, board, i):
            continue

        rx, ry, rz = board_energy(corners, board, params)
        energy = board.energy[ry][rx][rz]
        if energy > -6.0:
            for jj in range(3):
                for ii in range(3):
                    used[board.idx[jj][ii]] = 0
            continue

        # Grow loop
        while True:
            num_before = board.num
            for d in range(6 if params.corner_type == CornerType.MonkeySaddlePoint else 4):
                gtype, proposal = grow_board(corners, used, board, d, params)
                if gtype.value == 0:  # Failure
                    continue

                # Wrap proposal as mutable list for filter_board
                prop_list = list(proposal)
                eng_ref = [energy]
                filter_board(corners, used, board, prop_list, eng_ref, params)
                energy = eng_ref[0]

                if gtype.value == 1:  # Inside – retry this direction
                    d -= 1

            if board.num == num_before:
                break

        # Overlap handling
        if not params.overlay:
            boards.append(board)
            continue

        # Check overlap with existing boards
        overlap_indices = []
        for jb in range(len(boards)):
            overlapped = False
            for k1 in range(len(board.idx)):
                for k2 in range(len(board.idx[0])):
                    if board.idx[k1][k2] < 0:
                        continue
                    for l1 in range(len(boards[jb].idx)):
                        for l2 in range(len(boards[jb].idx[0])):
                            if board.idx[k1][k2] == boards[jb].idx[l1][l2]:
                                overlapped = True
                                break
                        if overlapped:
                            break
                    if overlapped:
                        break
                if overlapped:
                    break
            if overlapped:
                rx2, ry2, rz2 = board_energy(corners, boards[jb], params)
                ej = boards[jb].energy[ry2][rx2][rz2]
                overlap_indices.append((jb, ej))

        if not overlap_indices:
            boards.append(board)
        else:
            is_better = all(ej > energy for _, ej in overlap_indices)
            if is_better:
                surviving = [b for j, b in enumerate(boards)
                             if j not in oi for oi in overlap_indices for (_, _) in [(oi,)]]
                # Simpler: rebuild excluding removed indices
                rm_set = set(jb for jb, _ in overlap_indices)
                boards[:] = [b for j, b in enumerate(boards) if j not in rm_set]
                boards.append(board)

        if params.overlay:
            # Reset used flags for overlay mode
            used = [0] * len(corners.p)
            n += 2

    return boards
