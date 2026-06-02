"""
pycbdetect - Pure Python checkerboard/deltille pattern detection library.

Ported from libcbdetect (C++) to pure Python with NumPy dependency only.
Supports Python 3.9+.
"""

from pycbdetect.config import (
    DetectMethod,
    CornerType,
    GrowType,
    Params,
    Corner,
    Board,
)
from pycbdetect.find_corners import find_corners
from pycbdetect.boards_from_corners import boards_from_corners
from pycbdetect.plot_utils import plot_corners, plot_boards

__version__ = "0.1.0"

__all__ = [
    "DetectMethod",
    "CornerType",
    "GrowType",
    "Params",
    "Corner",
    "Board",
    "find_corners",
    "boards_from_corners",
    "plot_corners",
    "plot_boards",
]
