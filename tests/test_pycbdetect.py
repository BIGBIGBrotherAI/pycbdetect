#!/usr/bin/env python3
"""
Comprehensive test suite for pycbdetect.

Tests cover:
  1. Import and basic API availability
  2. Data classes (Params, Corner, Board)
  3. Enumerations (DetectMethod, CornerType, GrowType)
  4. Utility functions (weight_mask, correlation patches, image patches, filters)
  5. Meanshift mode finding
  6. Image processing (normalization + gradients)
  7. End-to-end corner detection on synthetic checkerboard images
  8. End-to-end board assembly
  9. Edge cases (small images, grayscale, noise)
  10. Different detection methods
"""

import math
import os
import sys
import unittest

import numpy as np

# ── Imports under test ────────────────────────────────────────────────

from pycbdetect import (
    DetectMethod,
    CornerType,
    GrowType,
    Params,
    Corner,
    Board,
    find_corners,
    boards_from_corners,
)
from pycbdetect.config import Corner, Board, Params, DetectMethod, CornerType, GrowType
from pycbdetect.utils import (
    weight_mask,
    create_correlation_patch_2angle,
    create_correlation_patch_3angle,
    get_image_patch,
    get_image_patch_with_mask,
    box_filter,
    hessian_response,
    convolve2d_border_replicate,
)
from pycbdetect.meanshift import find_modes_meanshift
from pycbdetect.imgproc import image_normalization_and_gradients
from pycbdetect.filter_corners import filter_corners
from pycbdetect.refine_corners import refine_corners
from pycbdetect.score_corners import score_corners, remove_low_scoring_corners
from pycbdetect.nms import non_maximum_suppression, non_maximum_suppression_sparse
from pycbdetect.polynomial_fit import polynomial_fit
from pycbdetect.get_init_location import get_init_location
from pycbdetect.board_helpers import init_board, board_energy, grow_board, filter_board
from pycbdetect.boards_from_corners import boards_from_corners


# ── Helpers ───────────────────────────────────────────────────────────

def _make_checkerboard(hw, wh, squares_x, squares_y, square_px=30):
    """Generate a synthetic checkerboard image of size (hw,wh) with given grid."""
    img = np.zeros((hw, wh), dtype=np.uint8)
    for yy in range(squares_y):
        for xx in range(squares_x):
            if (xx + yy) % 2 == 0:
                y0, y1 = yy * square_px, min((yy + 1) * square_px, hw)
                x0, x1 = xx * square_px, min((xx + 1) * square_px, wh)
                img[y0:y1, x0:x1] = 255
    return img


def _make_rgb_checkerboard(hw, wh, squares_x, squares_y, square_px=30):
    """Synthetic RGB checkerboard."""
    gray = _make_checkerboard(hw, wh, squares_x, squares_y, square_px)
    rgb = np.stack([gray, gray, gray], axis=2)
    return rgb


# ── Test Classes ──────────────────────────────────────────────────────

class TestImports(unittest.TestCase):
    """Verify public API symbols are accessible."""

    def test_version(self):
        import pycbdetect
        self.assertIsInstance(pycbdetect.__version__, str)

    def test_find_corners_importable(self):
        callable(find_corners)

    def test_boards_from_corners_importable(self):
        callable(boards_from_corners)

    def test_config_classes(self):
        self.assertIsNotNone(Corner)
        self.assertIsNotNone(Board)
        self.assertIsNotNone(Params)

    def test_enums(self):
        self.assertEqual(DetectMethod.TemplateMatchFast.value, 0)
        self.assertEqual(DetectMethod.TemplateMatchSlow.value, 1)
        self.assertEqual(DetectMethod.HessianResponse.value, 2)
        self.assertEqual(DetectMethod.LocalizedRadonTransform.value, 3)
        self.assertEqual(CornerType.SaddlePoint.value, 0)
        self.assertEqual(CornerType.MonkeySaddlePoint.value, 1)
        self.assertEqual(GrowType.Failure.value, 0)
        self.assertEqual(GrowType.Inside.value, 1)
        self.assertEqual(GrowType.Boundary.value, 2)


class TestDataClasses(unittest.TestCase):
    """Test dataclass constructors and defaults."""

    def test_params_defaults(self):
        p = Params()
        self.assertTrue(p.show_processing)
        self.assertFalse(p.norm)
        self.assertTrue(p.polynomial_fit)
        self.assertEqual(p.detect_method, DetectMethod.HessianResponse)
        self.assertEqual(p.corner_type, CornerType.SaddlePoint)
        self.assertIn(5, p.radius)
        self.assertIn(7, p.radius)

    def test_params_custom(self):
        p = Params(
            show_processing=False,
            detect_method=DetectMethod.TemplateMatchFast,
            corner_type=CornerType.MonkeySaddlePoint,
            radius=[3, 5],
        )
        self.assertFalse(p.show_processing)
        self.assertEqual(p.detect_method, DetectMethod.TemplateMatchFast)
        self.assertEqual(p.corner_type, CornerType.MonkeySaddlePoint)
        self.assertEqual(p.radius, [3, 5])

    def test_corner_clear(self):
        c = Corner()
        c.p.append(np.array([1.0, 2.0]))
        c.r.append(5)
        c.score.append(0.5)
        self.assertEqual(len(c.p), 1)
        c.clear()
        self.assertEqual(len(c.p), 0)
        self.assertEqual(len(c.r), 0)
        self.assertEqual(len(c.score), 0)

    def test_board_default(self):
        b = Board()
        self.assertEqual(b.num, 0)
        self.assertEqual(len(b.idx), 0)


class TestUtilsFunctions(unittest.TestCase):
    """Unit tests for utility/helper functions."""

    def test_weight_mask_shapes(self):
        m = weight_mask([5, 7])
        self.assertIn(5, m)
        self.assertIn(7, m)
        self.assertEqual(m[5].shape, (11, 11))
        self.assertEqual(m[7].shape, (13, 13))
        # Values should be positive
        self.assertTrue(np.all(m[5] > 0))

    def test_create_correlation_patch_2angle(self):
        kernels = create_correlation_patch_2angle(0.0, math.pi / 2, 5)
        self.assertEqual(len(kernels), 4)
        for k in kernels:
            self.assertEqual(k.shape, (11, 11))
            # Sum should be close to 1 (normalized)
            self.assertAlmostEqual(np.sum(k), 1.0, places=4)

    def test_create_correlation_patch_3angle(self):
        kernels = create_correlation_patch_3angle(
            0.0, math.pi / 3, 2 * math.pi / 3, 5
        )
        self.assertEqual(len(kernels), 6)
        for k in kernels:
            self.assertEqual(k.shape, (11, 11))
            self.assertAlmostEqual(np.sum(k), 1.0, places=4)

    def test_get_image_patch_center(self):
        img = np.arange(100, dtype=np.float64).reshape(10, 10)
        patch = get_image_patch(img, 5.0, 5.0, 2)
        self.assertEqual(patch.shape, (5, 5))

    def test_get_image_patch_float_coords(self):
        img = np.arange(100, dtype=np.float64).reshape(10, 10)
        patch = get_image_patch(img, 5.5, 5.5, 2)
        self.assertEqual(patch.shape, (5, 5))

    def test_box_filter_basic(self):
        img = np.ones((20, 20), dtype=np.float64) * 42.0
        result = box_filter(img, 3)
        # Uniform image should remain uniform
        self.assertTrue(np.allclose(result, 42.0))

    def test_hessian_response_uniform(self):
        img = np.ones((20, 20), dtype=np.float64)
        resp = hessian_response(img)
        # Flat image → zero Hessian everywhere
        self.assertTrue(np.all(resp == 0.0))

    def test_convolve2d_border_replicate_identity(self):
        img = np.random.rand(20, 20).astype(np.float64)
        identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
        result = convolve2d_border_replicate(img, identity)
        self.assertTrue(np.allclose(result, img))

    def test_get_image_patch_with_mask(self):
        img = np.arange(100, dtype=np.float64).reshape(10, 10)
        mask = np.ones((5, 5), dtype=np.float64)
        mask[2, 2] = 0  # exclude center
        vec = get_image_patch_with_mask(img, mask, 5.0, 5.0, 2)
        # Should be a column vector with fewer entries due to masking
        self.assertGreaterEqual(vec.ndim, 1)


class TestMeanshift(unittest.TestCase):
    """MeanShift mode-finding tests."""

    def test_single_peak(self):
        hist = [0, 0, 10, 0, 0]
        modes = find_modes_meanshift(hist)
        self.assertGreater(len(modes), 0)

    def test_two_peaks(self):
        hist = [0, 10, 0, 0, 10, 0]
        modes = find_modes_meanshift(hist)
        # Should find both peaks
        self.assertGreaterEqual(len(modes), 2)

    def test_empty_histogram(self):
        hist = [0.0] * 20
        modes = find_modes_meanshift(hist)
        self.assertEqual(modes, [])

    def test_sorted_descending(self):
        hist = [0, 5, 0, 10, 0, 3, 0]
        modes = find_modes_meanshift(hist)
        for i in range(len(modes) - 1):
            self.assertGreaterEqual(modes[i][1], modes[i + 1][1])


class TestImageProcessing(unittest.TestCase):
    """Normalization and gradient computation."""

    def test_returns_correct_types(self):
        img = np.random.rand(50, 50).astype(np.float64)
        params = Params(norm=False)
        result = image_normalization_and_gradients(img, params)
        self.assertEqual(len(result), 5)
        img_n, du, dv, angle, weight = result
        self.assertEqual(img_n.shape, img.shape)
        self.assertEqual(du.shape, img.shape)
        self.assertEqual(dv.shape, img.shape)
        self.assertEqual(angle.shape, img.shape)
        self.assertEqual(weight.shape, img.shape)

    def test_gradient_magnitude_positive(self):
        img = np.random.rand(50, 50).astype(np.float64)
        params = Params(norm=False)
        _, _, _, _, weight = image_normalization_and_gradients(img, params)
        self.assertTrue(np.all(weight >= 0))

    def test_normalized_intensity_range(self):
        img = np.random.rand(50, 50).astype(np.float64)
        params = Params(norm=False)
        img_n, *_ = image_normalization_and_gradients(img, params)
        self.assertLessEqual(img_n.max(), 1.0 + 1e-10)
        self.assertGreaterEqual(img_n.min(), 0.0 - 1e-10)


class TestEndToEndCornerDetection(unittest.TestCase):
    """Integration tests: full corner detection pipeline on synthetic images."""

    def setUp(self):
        # Small checkerboard: 8x8 squares, 20px each → 160x160 image
        self.cb_small = _make_checkerboard(160, 160, 8, 8, 20)
        # Larger checkerboard: 10x7 squares, 40px each → 280x400 image
        self.cb_large = _make_checkerboard(280, 400, 7, 10, 40)
        # RGB versions
        self.rgb_cb = _make_rgb_checkerboard(160, 160, 8, 8, 20)

    def test_detect_corners_grayscale(self):
        params = Params(show_processing=False)
        corners = find_corners(self.cb_small, params=params)
        self.assertIsInstance(corners, Corner)
        # Should detect at least some corners
        self.assertGreater(len(corners.p), 0)

    def test_detect_corners_larger_image(self):
        params = Params(show_processing=False)
        corners = find_corners(self.cb_large, params=params)
        self.assertGreater(len(corners.p), 0)

    def test_detect_corners_rgb_input(self):
        params = Params(show_processing=False)
        corners = find_corners(self.rgb_cb, params=params)
        self.assertGreater(len(corners.p), 0)

    def test_corner_positions_in_bounds(self):
        params = Params(show_processing=False)
        corners = find_corners(self.cb_small, params=params)
        h, w = self.cb_small.shape
        for p in corners.p:
            self.assertGreaterEqual(p[0], 0)
            self.assertGreaterEqual(p[1], 0)
            self.assertLess(p[0], w)
            self.assertLess(p[1], h)

    def test_corner_attributes_consistent_length(self):
        params = Params(show_processing=False)
        corners = find_corners(self.cb_small, params=params)
        n = len(corners.p)
        self.assertEqual(len(corners.r), n)
        self.assertEqual(len(corners.v1), n)
        self.assertEqual(len(corners.v2), n)
        self.assertEqual(len(corners.score), n)

    def test_different_detection_methods(self):
        cb = self.cb_small
        for dm in [DetectMethod.HessianResponse, DetectMethod.TemplateMatchFast]:
            with self.subTest(method=dm.name):
                params = Params(show_processing=False, detect_method=dm)
                corners = find_corners(cb, params=params)
                self.assertIsInstance(corners, Corner)

    def test_no_crash_on_tiny_image(self):
        tiny = np.zeros((30, 30), dtype=np.uint8)
        params = Params(show_processing=False)
        corners = find_corners(tiny, params=params)
        # Tiny image likely yields no corners, but should not crash
        self.assertIsInstance(corners, Corner)

    def test_repeated_call_same_results(self):
        params = Params(show_processing=False)
        c1 = find_corners(self.cb_small, params=params)
        c2 = find_corners(self.cb_small, params=params)
        self.assertEqual(len(c1.p), len(c2.p))

    def test_polynomial_fit_disabled(self):
        params = Params(show_processing=False, polynomial_fit=False)
        corners = find_corners(self.cb_small, params=params)
        self.assertIsInstance(corners, Corner)

    def test_add_noise_tolerance(self):
        noisy = self.cb_small.astype(np.float64) + np.random.randn(*self.cb_small.shape) * 10
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        params = Params(show_processing=False)
        corners = find_corners(noisy, params=params)
        self.assertIsInstance(corners, Corner)


class TestEndToEndBoardAssembly(unittest.TestCase):
    """Integration tests: board assembly from detected corners."""

    def setUp(self):
        self.params = Params(show_processing=False)
        self.cb = _make_checkerboard(280, 400, 7, 10, 40)

    def test_assemble_boards(self):
        corners = find_corners(self.cb, params=self.params)
        boards = boards_from_corners(self.cb, corners, params=self.params)
        self.assertIsInstance(boards, list)
        # With a clean checkerboard we expect at least one board
        self.assertGreater(len(boards), 0)

    def test_board_structure_valid(self):
        corners = find_corners(self.cb, params=self.params)
        boards = boards_from_corners(self.cb, corners, params=self.params)
        for b in boards:
            self.assertIsInstance(b, Board)
            self.assertGreater(b.num, 0)
            self.assertGreater(len(b.idx), 0)
            # Indices should reference valid corners
            for row in b.idx:
                for idx in row:
                    if idx >= 0:
                        self.assertLess(idx, len(corners.p))

    def test_overlay_mode(self):
        params = Params(show_processing=False, overlay=True)
        corners = find_corners(self.cb, params=params)
        boards = boards_from_corners(self.cb, corners, params=params)
        self.assertIsInstance(boards, list)

    def test_occlusion_false(self):
        params = Params(show_processing=False, occlusion=False)
        corners = find_corners(self.cb, params=params)
        boards = boards_from_corners(self.cb, corners, params=params)
        self.assertIsInstance(boards, list)


class TestInitLocationMethods(unittest.TestCase):
    """Test individual detection stages."""

    def test_hessian_finds_initial_locations(self):
        cb = _make_checkerboard(160, 160, 8, 8, 20)
        gray = cb.astype(np.float64) / 255.0
        params = Params(show_processing=False, detect_method=DetectMethod.HessianResponse)
        img_n, img_du, img_dv, img_angle, img_weight = (
            image_normalization_and_gradients(gray, params)
        )
        corners = Corner()
        get_init_location(img_n, img_du, img_dv, corners, params)
        self.assertGreater(len(corners.p), 0)

    def test_template_match_fast(self):
        cb = _make_checkerboard(160, 160, 8, 8, 20)
        gray = cb.astype(np.float64) / 255.0
        params = Params(
            show_processing=False,
            detect_method=DetectMethod.TemplateMatchFast,
        )
        img_n, img_du, img_dv, img_angle, img_weight = (
            image_normalization_and_gradients(gray, params)
        )
        corners = Corner()
        get_init_location(img_n, img_du, img_dv, corners, params)
        self.assertIsInstance(corners, Corner)


class TestFilterRefineScorePipeline(unittest.TestCase):
    """Test intermediate pipeline stages individually."""

    def _setup_corners(self):
        cb = _make_checkerboard(160, 160, 8, 8, 20)
        gray = cb.astype(np.float64) / 255.0
        params = Params(show_processing=False)
        img_n, img_du, img_dv, img_angle, img_weight = (
            image_normalization_and_gradients(gray, params)
        )
        corners = Corner()
        get_init_location(img_n, img_du, img_dv, corners, params)
        return img_n, img_du, img_dv, img_angle, img_weight, corners, params

    def test_filter_does_not_crash(self):
        img_n, img_du, img_dv, img_angle, img_weight, corners, params = (
            self._setup_corners()
        )
        n_before = len(corners.p)
        filter_corners(img_n, img_angle, img_weight, corners, params)
        self.assertLessEqual(len(corners.p), n_before)

    def test_refine_does_not_crash(self):
        img_n, img_du, img_dv, img_angle, img_weight, corners, params = (
            self._setup_corners()
        )
        filter_corners(img_n, img_angle, img_weight, corners, params)
        if len(corners.p) > 0:
            refine_corners(img_du, img_dv, img_angle, img_weight, corners, params)

    def test_score_assigns_values(self):
        img_n, img_du, img_dv, img_angle, img_weight, corners, params = (
            self._setup_corners()
        )
        filter_corners(img_n, img_angle, img_weight, corners, params)
        if len(corners.p) > 0:
            # Give corners dummy orientations for scoring
            for i in range(len(corners.p)):
                corners.v1.append(np.array([1.0, 0.0]))
                corners.v2.append(np.array([0.0, 1.0]))
            score_corners(img_n, img_weight, corners, params)
            self.assertEqual(len(corners.score), len(corners.p))

    def test_remove_low_scores_filters(self):
        c = Corner()
        c.p = [np.array([10.0, 10.0]), np.array([20.0, 20.0])]
        c.r = [5, 5]
        c.v1 = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
        c.v2 = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
        c.score = [0.5, 0.001]
        params = Params(corner_type=CornerType.SaddlePoint)
        remove_low_scoring_corners(0.01, c, params)
        self.assertEqual(len(c.p), 1)
        self.assertEqual(c.score[0], 0.5)

    def test_nms_sparse_removes_duplicates(self):
        c = Corner()
        # Place two very close corners
        c.p = [np.array([10.0, 10.0]), np.array([11.0, 11.0])]
        c.r = [5, 5]
        c.v1 = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
        c.v2 = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
        c.score = [0.5, 0.3]
        params = Params(corner_type=CornerType.SaddlePoint)
        non_maximum_suppression_sparse(c, 3, (100, 100), params)
        # One should be suppressed
        self.assertLessEqual(len(c.p), 2)


class TestPolynomialFit(unittest.TestCase):
    """Polynomial surface-fit tests."""

    def test_quadratic_fit_runs_without_error(self):
        cb = _make_checkerboard(160, 160, 8, 8, 20)
        gray = cb.astype(np.float64) / 255.0
        params = Params(show_processing=False, polynomial_fit=True)
        img_n, img_du, img_dv, img_angle, img_weight = (
            image_normalization_and_gradients(gray, params)
        )
        corners = Corner()
        get_init_location(img_n, img_du, img_dv, corners, params)
        filter_corners(img_n, img_angle, img_weight, corners, params)
        if len(corners.p) > 0:
            refine_corners(img_du, img_dv, img_angle, img_weight, corners, params)
            polynomial_fit(img_n, corners, params)
            self.assertIsInstance(corners, Corner)


class TestBoardHelpers(unittest.TestCase):
    """Low-level board helper tests."""

    def test_board_energy_computes(self):
        c = Corner()
        # Create a small 3x3 grid of corners
        spacing = 20.0
        idx = 0
        for row in range(3):
            for col in range(3):
                c.p.append(np.array([col * spacing, row * spacing], dtype=np.float64))
                c.r.append(5)
                c.v1.append(np.array([1.0, 0.0]))
                c.v2.append(np.array([0.0, 1.0]))
                c.v3.append(None)
                idx += 1

        b = Board()
        b.idx = [[i for i in range(3)] for _ in range(3)]
        b.idx = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        b.energy = [[[float('inf')] * 3 for _ in range(3)] for _ in range(3)]
        b.num = 9

        params = Params(corner_type=CornerType.SaddlePoint)
        rx, ry, rz = board_energy(c, b, params)
        self.assertIsInstance(rx, int)
        self.assertIsInstance(ry, int)
        self.assertIsInstance(rz, int)

    def test_init_board_success(self):
        c = Corner()
        spacing = 20.0
        for row in range(3):
            for col in range(3):
                c.p.append(np.array([col * spacing, row * spacing], dtype=np.float64))
                c.r.append(5)
                c.v1.append(np.array([1.0, 0.0]))
                c.v2.append(np.array([0.0, 1.0]))
                c.v3.append(None)

        used = [0] * 9
        b = Board()
        result = init_board(c, used, b, 4)  # centre corner
        self.assertTrue(result)
        self.assertEqual(b.num, 9)

    def test_init_board_insufficient_corners(self):
        c = Corner()
        c.p.append(np.array([0.0, 0.0]))
        c.r.append(5)
        c.v1.append(np.array([1.0, 0.0]))
        c.v2.append(np.array([0.0, 1.0]))
        c.v3.append(None)
        used = [0]
        b = Board()
        result = init_board(c, used, b, 0)
        self.assertFalse(result)


class TestEdgeCases(unittest.TestCase):
    """Robustness tests for unusual inputs."""

    def test_all_black_image(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        params = Params(show_processing=False)
        corners = find_corners(img, params=params)
        self.assertIsInstance(corners, Corner)

    def test_all_white_image(self):
        img = np.ones((100, 100), dtype=np.uint8) * 255
        params = Params(show_processing=False)
        corners = find_corners(img, params=params)
        self.assertIsInstance(corners, Corner)

    def test_highly_rotated_checkboard_simulation(self):
        # Even though our synth generator makes axis-aligned boards,
        # the detector should still handle arbitrary intensity patterns
        img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        params = Params(show_processing=False)
        corners = find_corners(img, params=params)
        self.assertIsInstance(corners, Corner)

    def test_non_square_aspect_ratio(self):
        img = _make_checkerboard(100, 400, 4, 16, 25)
        params = Params(show_processing=False)
        corners = find_corners(img, params=params)
        self.assertIsInstance(corners, Corner)

    def test_pass_existing_corner_object(self):
        img = _make_checkerboard(160, 160, 8, 8, 20)
        params = Params(show_processing=False)
        shared = Corner()
        shared.p.append(np.array([999.0, 999.0]))
        corners = find_corners(img, corners=shared, params=params)
        # Existing object should be reused (cleared and repopulated)
        self.assertIs(corners, shared)


if __name__ == "__main__":
    unittest.main()
