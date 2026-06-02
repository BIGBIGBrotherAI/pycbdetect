"""
MeanShift mode-finding algorithm for histograms.

Efficient mean-shift approximation by histogram smoothing.
"""

import math

import numpy as np
from collections import OrderedDict


def find_modes_meanshift(hist, sigma=1.5):
    """Find modes of a histogram via Gaussian-smoothed MeanShift.

    Args:
        hist: 1D array of histogram counts/values
        sigma: bandwidth for Gaussian smoothing

    Returns:
        Sorted list of (index, smoothed_value) tuples sorted descending by value.
    """
    hist = np.asarray(hist, dtype=np.float64)
    n = len(hist)

    r = int(round(2 * sigma))
    weight = np.exp(-0.5 * np.arange(2 * r + 1) ** 2 / sigma ** 2) / math.sqrt(2 * math.pi) / sigma

    # Smoothed histogram (circular convolution)
    hist_smoothed = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(2 * r + 1):
            hist_smoothed[(i + r) % n] += hist[(i + j) % n] * weight[j]

    # Check if anything significant
    if np.max(hist_smoothed) < 1e-6:
        return []

    # Mode finding via greedy hill-climbing
    visited = np.zeros(n, dtype=bool)
    hash_table = OrderedDict()

    for i in range(n):
        j = i
        if not visited[j]:
            while True:
                visited[j] = True
                j1 = (j + 1) % n
                j2 = (j + n - 1) % n
                h0 = hist_smoothed[j]
                h1 = hist_smoothed[j1]
                h2 = hist_smoothed[j2]
                if h1 >= h0 and h1 >= h2:
                    j = j1
                elif h2 > h0 and h2 > h1:
                    j = j2
                else:
                    break
            hash_table[j] = hist_smoothed[j]

    # Sort by value descending
    modes = sorted(hash_table.items(), key=lambda x: x[1], reverse=True)
    return modes
