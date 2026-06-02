#!/usr/bin/env python3
"""
Quick smoke test for pycbdetect — runs in seconds.

Only exercises fast unit-level checks and a minimal end-to-end pass
with a tiny synthetic checkerboard.
"""

import math
import sys

import numpy as np

PASS_COUNT = 0
FAIL_COUNT = 0

def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✓ {name}")
    else:
        FAIL_COUNT += 1
        msg = f"  ✗ {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)

# ── 1. Basic imports ──────────────────────────────────────────────────
print("\n=== 1. Imports ===")
try:
    from pycbdetect import (
        Params, Corner, Board,
        DetectMethod, CornerType, GrowType,
        find_corners, boards_from_corners,
    )
    from pycbdetect.utils import (
        weight_mask, create_correlation_patch_2angle,
        get_image_patch, box_filter, hessian_response,
    )
    from pycbdetect.meanshift import find_modes_meanshift
    check("Public API imported", True)
except Exception as e:
    check("Public API imported", False, str(e))
    sys.exit(1)

check("__version__", isinstance(__import__("pycbdetect").__version__, str))

# ── 2. Enums ──────────────────────────────────────────────────────────
print("\n=== 2. Enums ===")
check("DetectMethod values",
      DetectMethod.TemplateMatchFast.value == 0 and
      DetectMethod.HessianResponse.value == 2)
check("CornerType values",
      CornerType.SaddlePoint.value == 0 and
      CornerType.MonkeySaddlePoint.value == 1)
check("GrowType values",
      GrowType.Failure.value == 0 and
      GrowType.Boundary.value == 2)

# ── 3. Data classes ───────────────────────────────────────────────────
print("\n=== 3. Data classes ===")
p = Params()
check("Params defaults", p.show_processing and p.polynomial_fit)
check("Params radius", 5 in p.radius and 7 in p.radius)

c = Corner()
c.p.append(np.array([1.0, 2.0])); c.r.append(5); c.score.append(.5)
check("Corner append", len(c.p) == 1)
c.clear()
check("Corner clear", len(c.p) == 0)

b = Board()
check("Board default num==0", b.num == 0)

# ── 4. Utils ──────────────────────────────────────────────────────────
print("\n=== 4. Utilities ===")
m = weight_mask([5])
check("weight_mask shape", m[5].shape == (11, 11))

ker = create_correlation_patch_2angle(0, math.pi/2, 5)
check("corr_patch_2angle count", len(ker) == 4)
check("corr_patch_2angle normalize", all(abs(sum(k.flatten())) - 1 < 1e-4 for k in ker))

patch = get_image_patch(np.eye(10, dtype=float), 5.0, 5.0, 2)
check("get_image_patch shape", patch.shape == (5, 5))

bf = box_filter(np.ones((20, 20), dtype=float) * 42, 3)
check("box_filter constant", np.allclose(bf, 42.0))

hr = hessian_response(np.ones((20, 20), dtype=float))
check("hessian flat image", np.all(hr == 0))

# ── 5. Meanshift ─────────────────────────────────────────────────────
print("\n=== 5. Meanshift ===")
modes = find_modes_meanshift([0, 10, 0, 0, 10, 0])
check("two peaks found", len(modes) >= 2)
empty_modes = find_modes_meanshift([0.0] * 20)
check("empty histogram", empty_modes == [])

# ── 6. Fast end-to-end (tiny checkerboard) ───────────────────────────
print("\n=== 6. Mini end-to-end ===")

# 4×4 checkerboard, 10px squares → 40×40 image
sq = 10
hx, hy = 4, 4
W, H = hx * sq, hy * sq
img = np.zeros((H, W), dtype=np.uint8)
for yy in range(hy):
    for xx in range(hx):
        if (xx + yy) % 2 == 0:
            img[yy*sq:(yy+1)*sq, xx*sq:(xx+1)*sq] = 255

params = Params(show_processing=False, polynomial_fit=False)

try:
    corners = find_corners(img, params=params)
    check("find_corners returns Corner", isinstance(corners, Corner))
    check("some corners detected", len(corners.p) > 0, f"{len(corners.p)} corners")
    if len(corners.p) > 0:
        check("positions in bounds",
              all(0 <= p[0] < W and 0 <= p[1] < H for p in corners.p))
        check("consistent lengths",
              len(corners.p) == len(corners.r) == len(corners.score))
except Exception as e:
    check("find_corners no exception", False, str(e))

try:
    boards = boards_from_corners(img, corners, params=params)
    check("boards_from_corners returns list", isinstance(boards, list))
    if boards:
        check("at least one board", True, f"{len(boards)} board(s)")
        check("board has cells", boards[0].num > 0, f"num={boards[0].num}")
except Exception as e:
    check("boards_from_corners no exception", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────
total = PASS_COUNT + FAIL_COUNT
print(f"\n{'━'*40}")
print(f"Results: {PASS_COUNT}/{total} passed, {FAIL_COUNT} failed")
if Fail_COUNT := FAIL_COUNT:
    sys.exit(1)
else:
    print("All smoke tests passed!")
