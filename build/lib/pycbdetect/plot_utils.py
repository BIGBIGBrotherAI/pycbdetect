"""
Visualization utilities for plotting detected corners and boards.

Requires matplotlib (lazy-imported so it remains an optional dependency).
"""

import numpy as np


def plot_corners(img, corners, title="Corners"):
    """Overlay red circles on corner positions.

    Args:
        img: Colour or grayscale image (numpy array).
        corners: Either a Corner object or a list of (x,y) arrays/lists.
        title: Window title (ignored if headless backend).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required for visualization")

    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        cmap = 'gray'
    else:
        cmap = None

    fig, ax = plt.subplots(figsize=(10, 7))
    if cmap:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)

    if hasattr(corners, 'p'):
        pts = corners.p
    else:
        pts = corners

    for pt in pts:
        x, y = float(pt[0]), float(pt[1])
        circ = plt.Circle((x, y), 5, color='red', fill=False, linewidth=1.5)
        ax.add_artist(circ)

    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_boards(img, corners, boards, title="Checkerboards"):
    """Visualise detected checkerboard patterns.

    Draws green dots for board corners connected by blue edges, plus
    labels showing internal corner IDs.

    Args:
        img: Source image.
        corners: Detected Corner container.
        boards: List of Board objects.
        title: Figure title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required for visualization")

    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        cmap = 'gray'
    else:
        cmap = None

    fig, ax = plt.subplots(figsize=(12, 8))
    if cmap:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)

    for bi, board in enumerate(boards):
        # Draw connections
        rows = len(board.idx)
        cols = len(board.idx[0]) if rows > 0 else 0

        # Horizontal edges
        for r in range(rows):
            for c in range(cols - 1):
                ci = board.idx[r][c]
                cj = board.idx[r][c + 1]
                if ci >= 0 and cj >= 0:
                    p1 = corners.p[ci]
                    p2 = corners.p[cj]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=1)

        # Vertical edges
        for r in range(rows - 1):
            for c in range(cols):
                ci = board.idx[r][c]
                cj = board.idx[r + 1][c]
                if ci >= 0 and cj >= 0:
                    p1 = corners.p[ci]
                    p2 = corners.p[cj]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=1)

        # Mark corners
        for r in range(rows):
            for c in range(cols):
                ci = board.idx[r][c]
                if ci >= 0:
                    pt = corners.p[ci]
                    circ = plt.Circle((pt[0], pt[1]), 5, color='green', ec='darkgreen', lw=1)
                    ax.add_artist(circ)
                    ax.annotate(str(ci), (pt[0] - 10, pt[1] - 8), fontsize=7, color='yellow')

    ax.set_title(title)
    plt.tight_layout()
    plt.show()
