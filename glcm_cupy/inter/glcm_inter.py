from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List

import cupy as cp
import numpy as np
from skimage.util import view_as_windows

from glcm_cupy.conf import *
from glcm_cupy.glcm_base import GLCMBase


class Direction(Enum):
    EAST = 0
    SOUTH_EAST = 1
    SOUTH = 2
    SOUTH_WEST = 3


def glcm_inter(
    im: np.ndarray,
    step_size: int = 1,
    radius: int = 2,
    bin_from: int = 256,
    bin_to: int = 256,
    directions: List[Direction] = (Direction.EAST,
                                   Direction.SOUTH_EAST,
                                   Direction.SOUTH,
                                   Direction.SOUTH_WEST),
    max_partition_size: int = MAX_PARTITION_SIZE,
    max_threads: int = MAX_THREADS,
    normalize_features: bool = True
) -> np.ndarray:
    """
    Examples:
        To scale down the image from a 128 max value to 32, we use
        bin_from = 128, bin_to = 32.

        The range will collapse from 128 to 32.

        This thus optimizes the GLCM speed.

    Args:
        im: Image to Process
        step_size: Stride Between GLCMs
        radius: Radius of Window
        bin_from: Binarize from.
        bin_to: Binarize to.
        directions: Directions to pair the windows.
        max_partition_size: Maximum number of windows to parse at once
        max_threads: Maximum threads for CUDA
        normalize_features: Whether to normalize features to [0, 1]

    Returns:


    """
    return GLCMInter(step_size, radius, bin_from, bin_to,
                     max_partition_size, max_threads,
                     normalize_features, directions).run(im)


@dataclass
class GLCMInter(GLCMBase):
    directions: List[Direction] = (
        Direction.EAST,
        Direction.SOUTH_EAST,
        Direction.SOUTH,
        Direction.SOUTH_WEST
    )

    def glcm_cells(self, im: np.ndarray) -> float:
        shape = self.glcm_shape(im)
        return np.prod(shape) * len(self.directions)

    def glcm_shape(self, im: np.ndarray):
        """ Calculate the image shape after GLCM

        Returns:
            Shape of Image after GLCM
        """

        shape = im.shape
        if len(shape) < 2:
            raise ValueError(f"Image must be at least 2 dims.")

        return (shape[0] - 2 * self.step_size - 2 * self.radius,
                shape[1] - 2 * self.step_size - 2 * self.radius,
                *shape[2:])

    def _from_3dimage(self, im: np.ndarray) -> np.ndarray:
        """ Generates the GLCM from a multi band image

        Args:
            im: A 3 dim image as an ndarray

        Returns:
            The GLCM Array 4dim with shape
                rows, cols, channel, feature
        """

        glcm_chs = [
            self._from_2dimage(im[..., ch]) for ch in range(im.shape[-1])
        ]

        return np.stack(glcm_chs, axis=2)

    def make_windows(self, im: np.ndarray) -> \
        List[Tuple[np.ndarray, np.ndarray]]:
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

        Returns:
            A List of I, J windows based on the directions.

            For each window:
                The first dimension: xy flat window indexes,
                the last dimension: xy flat indexes within each window.

        """

        glcm_h, glcm_w, *_ = self.glcm_shape(im)
        if glcm_h <= 0 or glcm_w <= 0:
            raise ValueError(
                f"Step Size & Diameter exceeds size for windowing. "
                f"im.shape[0] {im.shape[0]} "
                f"- 2 * step_size {step_size} "
                f"- 2 * radius {radius} <= 0 or"
                f"im.shape[1] {im.shape[1]} "
                f"- 2 * step_size {step_size} "
                f"- 2 * radius {radius} + 1 <= 0 was not satisfied."
            )

        ij = cp.asarray(view_as_windows(im, (self._diameter, self._diameter)))

        ijs: List[Tuple[np.ndarray, np.ndarray]] = []

        for direction in self.directions:
            i, j = self.pair_windows(ij, direction=direction)

            i = i.reshape((-1, *i.shape[-2:])) \
                .reshape((i.shape[0] * i.shape[1], -1))
            j = j.reshape((-1, *j.shape[-2:])) \
                .reshape((j.shape[0] * j.shape[1], -1))
            ijs.append((i, j))

        return ijs

    def pair_windows(self, ij: np.ndarray, direction: Direction):
        """ Pairs the ij windows in specified direction

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
            direction: Direction to pair

        Returns:

        """
        step = self.step_size
        i = ij[step:-step, step:-step]
        if direction == Direction.EAST:
            j = ij[step:-step, step * 2:]
        elif direction == Direction.SOUTH:
            j = ij[step * 2:, step:-step]
        elif direction == Direction.SOUTH_WEST:
            j = ij[step * 2:, :-step * 2]
        elif direction == Direction.SOUTH_EAST:
            j = ij[step * 2:, step * 2:]
        else:
            raise ValueError("direction must be of class Direction.")

        return i, j
