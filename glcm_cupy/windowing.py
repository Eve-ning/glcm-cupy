from __future__ import annotations

from enum import Enum
from typing import Tuple, List, Iterable

import cupy as cp
import numpy as np
from skimage.util import view_as_windows


class Direction(Enum):
    EAST = 0
    SOUTH_EAST = 1
    SOUTH = 2
    SOUTH_WEST = 3


def make_windows(im: np.ndarray,
                 radius: int,
                 step_size: int,
                 directions: Iterable[Direction] =
                 (Direction.EAST,
                  Direction.SOUTH_EAST,
                  Direction.SOUTH,
                  Direction.SOUTH_WEST)
                 ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """ From a 2D image np.ndarray, convert it into GLCM IJ windows.

    Examples:

        Input 4 x 4 image. Radius = 1. Step Size = 1.

        +-+-+-+-+         +-----+
        |       |         | +---+-+
        |       |  ---->  | |   | |
        |       |         | |   | |
        |       |         +-+---+ |
        +-+-+-+-+           +-----+      flat
          4 x 4           1 x 1 x 3 x 3  ----> 1 x 9
                          +---+   +---+
                          flat    flat

        The output will be flattened on the x,y,

        Input 5 x 5 image. Radius = 1. Step Size = 1.

        1-2-+-+-+-+         1-----+    2-----+
        3 4       |         | +---+-+  | +---+-+  3-----+    4-----+
        |         |  ---->  | |   | |  | |   | |  | +---+-+  | +---+-+
        |         |         | |   | |  | |   | |  | |   | |  | |   | |
        |         |         +-+---+ |  +-+---+ |  | |   | |  | |   | |
        |         |           +-----+    +-----+  +-+---+ |  +-+---+ |
        +-+-+-+-+-+                                 +-----+    +-----+
          4 x 4                         2 x 2 x 3 x 3 ----> 4 x 9
                                        +---+   +---+ flat
                                        flat    flat
    Args:
        im: Input Image
        radius: Radius of Window
        step_size: Step Size between ij pairs
        directions: Directions to pair the windows.

    Returns:
        A List of I, J windows based on the directions.

        For each window:
            The first dimension: xy flat window indexes,
            the last dimension: xy flat indexes within each window.

    """

    # This will yield a shape (window_i, window_j, row, col)
    # E.g. 100x100 with 5x5 window -> 96, 96, 5, 5
    shape = im_shape_after_glcm(im.shape, step_size=step_size, radius=radius)
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(
            f"Step Size & Diameter exceeds size for windowing. "
            f"im.shape[0] {im.shape[0]} "
            f"- 2 * step_size {step_size} "
            f"- 2 * radius {radius} <= 0 or"
            f"im.shape[1] {im.shape[1]} "
            f"- 2 * step_size {step_size} "
            f"- 2 * radius {radius} + 1 <= 0 was not satisfied."
        )

    diameter = radius * 2 + 1
    ij = cp.asarray(view_as_windows(im, (diameter, diameter)))

    ijs: List[Tuple[np.ndarray, np.ndarray]] = []

    for direction in directions:
        i, j = pair_windows(ij, step_size=step_size, direction=direction)

        i = i.reshape((-1, i.shape[-2], i.shape[-1])) \
            .reshape((i.shape[0] * i.shape[1], -1))
        j = j.reshape((-1, j.shape[-2], j.shape[-1])) \
            .reshape((j.shape[0] * j.shape[1], -1))
        ijs.append((i, j))

    return ijs


def pair_windows(ij: np.ndarray,
                 step_size: int,
                 direction: Direction):
    """ Pairs the ij windows

    Notes:

        For an image:

        +-----------+
        |           |
        |  +-----+  |
        |  |     |  |
        |  |     |  |
        |  +-----+  |
        |           |
        +-----------+
                  <-> Distance = Step Size

        The i window will always be the one in the middle
        We move j window around the outer skirts to define the pair.

    Args:
        ij: The ij output from make_windows
        step_size: Step Size between pairs
        direction: Direction to pair

    Returns:

    """
    i = ij[step_size:-step_size, step_size:-step_size]
    if direction == Direction.EAST:
        j = ij[step_size:-step_size, step_size * 2:]
    elif direction == Direction.SOUTH:
        j = ij[step_size * 2:, step_size:-step_size]
    elif direction == Direction.SOUTH_WEST:
        j = ij[step_size * 2:, :-step_size * 2]
    elif direction == Direction.SOUTH_EAST:
        j = ij[step_size * 2:, step_size * 2:]
    else:
        raise ValueError("direction must be of class Direction.")

    return i, j


def im_shape_after_glcm(im_shape: Tuple,
                        step_size: int,
                        radius: int):
    """ Calculate the image shape after GLCM

    Args:
        im_shape: Image Shape before GLCM
        step_size: Step Size
        radius: Radius of GLCM

    Returns:
        Shape of Image after GLCM
    """

    if len(im_shape) < 2:
        raise ValueError(f"Image must be at least 2 dims.")

    return (im_shape[0] - 2 * step_size - 2 * radius,
            im_shape[1] - 2 * step_size - 2 * radius,
            *im_shape[2:])
