# pycbdetect

Pure-Python port of [libcbdetect](http://www.cvlibs.net/software/libcbdetect/) — fully automatic sub-pixel checkerboard / chessboard / deltille pattern detection for camera calibration.

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/github/license/ftdlyc/libcbdetect)
![Status](https://img.shields.io/badge/status-beta-yellow.svg)

## Overview

`pycbdetect` implements the complete corner detection and board assembly pipeline originally developed in C++, rewritten in pure Python with only **NumPy** and **SciPy** as runtime dependencies. This eliminates the need for compiling C++ extensions, enabling seamless deployment across platforms.

Supported patterns:
- **Checkerboard** (chessboard) — classic black-white alternating grid
- **Deltille** — triangular tessellation for robust calibration

## Features

- ✅ No compiled C++ extension — works anywhere Python 3.9+ runs
- ✅ Minimal runtime dependencies: **NumPy** + **SciPy** only
- ✅ Identical algorithms to the original C++ library:
  - Template Matching (fast / slow)
  - Hessian Response
  - Localized Radon Transform
  - Zero-crossing + angular-mode filtering
  - Iterative sub-pixel position refinement
  - Quadratic / cubic polynomial surface fitting
  - Correlation-based scoring + Non-Maximum Suppression (NMS)
  - Energy-driven board assembly with directional growth

## Installation

```bash
# From PyPI (when published)
pip install pycbdetect

# Or clone and install locally
git clone https://github.com/ftdlyc/libcbdetect.git
cd libcbdetect/pycbdetect
pip install .
```

For interactive visualization, install the optional `viz` extras:

```bash
pip install ".[viz]"
```

## Quick Start

### Step 1: Detect Corners

```python
import numpy as np
from pycbdetect import Params, find_corners

# Load image (uint8 RGB or grayscale)
img = np.imread("calibration_photo.jpg")

params = Params(show_processing=True)
corners = find_corners(img, params=params)

print(f"Detected {len(corners.p)} corners")
```

Each detected corner contains:
- `.p[i]` — sub-pixel position `(x, y)` as `np.ndarray`
- `.r[i]` — detection radius used
- `.v1[i]`, `.v2[i]` — estimated edge direction vectors
- `.score[i]` — quality score (higher is better)

### Step 2: Assemble Boards

```python
from pycbdetect import boards_from_corners

boards = boards_from_corners(img, corners, params=params)
print(f"Assembled {len(boards)} board(s)")

for b in boards:
    rows, cols = len(b.idx), len(b.idx[0])
    print(f"  Board: {rows}×{cols} grid, {b.num} occupied cells")
```

### Step 3: Visualize (Optional)

```python
from pycbdetect import plot_corners, plot_boards

plot_corners(img, corners, title="Detected Corners")
plot_boards(img, corners, boards, title="Assembled Boards")
```

*(Requires `matplotlib`; install via `pip install "pycbdetect[viz]"`)*

## Complete Workflow Example

```python
import numpy as np
from pycbdetect import (
    Params, DetectMethod, CornerType,
    find_corners, boards_from_corners,
    plot_corners, plot_boards,
)

# --- Configure parameters ---
params = Params(
    show_processing=True,          # Print pipeline progress
    detect_method=DetectMethod.HessianResponse,  # Initialization method
    corner_type=CornerType.SaddlePoint,           # Checkerboard mode
    polynomial_fit=True,                       # Enable sub-pixel refinement
    radius=[5, 7],                            # Multi-scale detection
    score_thr=0.01,                           # Quality threshold
)

# --- Load image ---
img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# --- Detect corners ---
corners = find_corners(img, params=params)
print(f"Corners: {len(corners.p)}")

# --- Assemble boards ---
boards = boards_from_corners(img, corners, params=params)
print(f"Boards: {len(boards)}")

# --- Visualize ---
plot_corners(img, corners)
plot_boards(img, corners, boards)
```

## API Reference

### Core Functions

| Function | Signature | Description |
|---|---|---|
| `find_corners` | `find_corners(img, corners=None, params=None)` | Execute full corner detection pipeline |
| `boards_from_corners` | `boards_from_corners(img, corners, boards=None, params=None)` | Group corners into structured boards |
| `plot_corners` | `plot_corners(img, corners, title="Corners")` | Display corner overlays (needs matplotlib) |
| `plot_boards` | `plot_boards(img, corners, boards, title="Boards")` | Display board overlays (needs matplotlib) |

### Data Structures

#### `Params` — Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `show_processing` | `bool` | `True` | Print pipeline stage progress to stderr |
| `norm` | `bool` | `False` | Apply image normalization preprocessing |
| `norm_half_kernel_size` | `int` | `31` | Half-kernel size for normalization filter |
| `polynomial_fit` | `bool` | `True` | Enable polynomial surface fitting for sub-pixel precision |
| `polynomial_fit_half_kernel_size` | `int` | `4` | Kernel size for polynomial fitting |
| `init_loc_thr` | `float` | `0.01` | Threshold for initial corner location acceptance |
| `score_thr` | `float` | `0.01` | Minimum corner quality score to retain |
| `strict_grow` | `bool` | `True` | Strict mode for board expansion |
| `overlay` | `bool` | `False` | Allow overlapping board hypotheses |
| `occlusion` | `bool` | `True` | Support partially occluded boards |
| `detect_method` | `DetectMethod` | `HessianResponse` | Method for initializing corner candidates |
| `corner_type` | `CornerType` | `SaddlePoint` | Target corner topology |
| `radius` | `List[int]` | `[5, 7]` | Radii for multi-scale detection |

#### `DetectMethod` — Initialization Strategy

| Value | Name | Description |
|---|---|---|
| `0` | `TemplateMatchFast` | Fast template matching (4 angle pairs) |
| `1` | `TemplateMatchSlow` | Exhaustive template matching (many angle pairs) |
| `2` | `HessianResponse` | Hessian determinant response (recommended) |
| `3` | `LocalizedRadonTransform` | Rotation-blur Radon transform |

#### `CornerType` — Corner Topology

| Value | Name | Description |
|---|---|---|
| `0` | `SaddlePoint` | Standard checkerboard corner (2 orthogonal edges) |
| `1` | `MonkeySaddlePoint` | Deltille corner (3 symmetrically arranged edges) |

#### `Corner` — Detected Corner Container

| Attribute | Type | Description |
|---|---|---|
| `p` | `List[np.ndarray]` | Position arrays `[x, y]` for each corner |
| `r` | `List[int]` | Radius associated with each detection |
| `v1` | `List[np.ndarray]` | First edge direction vector |
| `v2` | `List[np.ndarray]` | Second edge direction vector |
| `v3` | `List[np.ndarray]` | Third edge direction (deltille only) |
| `score` | `List[float]` | Quality score for each corner |

##### Methods
- `clear()` — Reset all attributes to empty lists

#### `Board` — Assembled Board Hypothesis

| Attribute | Type | Description |
|---|---|---|
| `idx` | `List[List[int]]` | 2D grid of corner indices (`-1` = unoccupied) |
| `energy` | `List[List[List[float]]]` | Per-cell structural energy tensor |
| `num` | `int` | Number of occupied cells |

### Internal Modules

Advanced users may import lower-level modules directly:

| Module | Purpose |
|---|---|
| `pycbdetect.imgproc` | Image normalization and gradient computation |
| `pycbdetect.get_init_location` | Individual initialization strategies |
| `pycbdetect.filter_corners` | Sector-alternation prefilter |
| `pycbdetect.refine_corners` | Orientiation estimation + Gauss-Newton relocation |
| `pycbdetect.polynomial_fit` | Cone-weighted surface fitting |
| `pycbdetect.score_corners` | Correlation scoring and thresholding |
| `pycbdetect.nms` | Dense and sparse NMS routines |
| `pycbdetect.meanshift` | Histogram mode-finding via MeanShift |
| `pycbdetect.board_helpers` | Board init, growth, energy, and filtering primitives |
| `pycbdetect.utils` | Low-level helpers (patches, masks, convolutions) |

## Testing

Install the `dev` extras and run the smoke test:

```bash
pip install ".[dev]"
cd pycbdetect
python smoke_test.py
```

Or run the full test suite:

```bash
python -m pytest tests/ -v
```

## Performance Notes

Being a pure-Python implementation, `pycbdetect` trades speed for convenience compared to the C++ original. Typical wall-clock times on modern hardware:

| Operation | Approximate Time |
|---|---|
| Corner detection (single image, 640×480) | 1–5 seconds |
| Board assembly | < 1 second |

Tips for reducing latency:
- Disable `polynomial_fit` if approximate positions suffice
- Reduce `radius` list length (fewer scales to evaluate)
- Use `DetectMethod.TemplateMatchFast` instead of `HessianResponse`

## Changelog

### v0.1.0 (2025-06)
- Initial release
- Full corner detection pipeline (init → filter → refine → score → NMS)
- Board assembly with energy-driven growth
- Dual-resolution merging (native + scaled pass)
- Visualization support via matplotlib

## Comparison with Alternatives

| Feature | pycbdetect | opencv.calibrateCamera | calibra_tools |
|---|---|---|---|
| Pure Python | ✅ | ❌ (OpenCV bindings) | ⚠️ Mixed |
| Deltille support | ✅ | ❌ | ❌ |
| Sub-pixel refinement | ✅ (iterative GN) | ✅ | ✅ |
| Occlusion tolerance | ✅ | Limited | Varies |
| Cross-platform deploy | ✅ (pip) | ⚠️ Binary wheels | ⚠️ |

## Contributing

Contributions welcome! Please submit pull requests to the repository:

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Commit changes (`git commit -am "Add my feature"`)
4. Push to origin (`git push origin feat/my-feature`)
5. Submit Pull Request

Before submitting, please run the smoke test and ensure all checks pass.

## References

1. Geiger et al., *"Automatic Camera and Range Sensor Calibration Using a Single Shot"*, ICRA 2012  
   http://www.cvlibs.net/publications/GeigerEtAl_ICRA2012.pdf
2. Schönbein et al., *"Calibrating and Centering Quasi-Central Catadioptric Cameras"*, ICRA 2014
3. Placht et al., *"ROCHEDE: Robust Checkerboard Advanced Detection"*, ECCV 2014
4. Ha et al., *"Deltille Grids for Geometric Camera Calibration"*, ICCV 2017
5. Duda & Frese, *"Accurate Detection and Localisation of Checkerboard Corners"*, BMVC 2018

## License

GNU GPL v3 or later — same license as the original [libcbdetect](http://www.cvlibs.net/software/libcbdetect/).

See [LICENSE](../LICENSE) for full terms.
