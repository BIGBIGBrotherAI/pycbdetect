"""
Configuration data structures for pycbdetect.

Contains Params, Corner, Board structs and enumeration types.
"""

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np


class DetectMethod(IntEnum):
    """Detection method for corner initialization."""
    TemplateMatchFast = 0
    TemplateMatchSlow = 1
    HessianResponse = 2
    LocalizedRadonTransform = 3


class CornerType(IntEnum):
    """Type of corner to detect."""
    SaddlePoint = 0
    MonkeySaddlePoint = 1


class GrowType(IntEnum):
    """Result type from board growing."""
    Failure = 0
    Inside = 1
    Boundary = 2


@dataclass
class Params:
    """Parameters controlling the detection behavior."""
    show_processing: bool = True
    show_debug_image: bool = False
    show_grow_processing: bool = False
    norm: bool = False
    polynomial_fit: bool = True
    norm_half_kernel_size: int = 31
    polynomial_fit_half_kernel_size: int = 4
    init_loc_thr: float = 0.01
    score_thr: float = 0.01
    strict_grow: bool = True
    overlay: bool = False
    occlusion: bool = True
    detect_method: DetectMethod = DetectMethod.HessianResponse
    corner_type: CornerType = CornerType.SaddlePoint
    radius: List[int] = field(default_factory=lambda: [5, 7])


@dataclass
class Corner:
    """Container for detected corners and their properties."""
    def __post_init__(self):
        self.p: List[np.ndarray] = []
        self.r: List[int] = []
        self.v1: List[np.ndarray] = []
        self.v2: List[np.ndarray] = []
        self.v3: List[np.ndarray] = []
        self.score: List[float] = []

    def clear(self):
        """Reset all fields."""
        self.p = []
        self.r = []
        self.v1 = []
        self.v2 = []
        self.v3 = []
        self.score = []


@dataclass
class Board:
    """Represents a detected checkerboard pattern."""
    idx: List[List[int]] = field(default_factory=list)
    energy: List[List[List[float]]] = field(default_factory=list)
    num: int = 0
