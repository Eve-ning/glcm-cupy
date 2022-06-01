from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Tuple, List

import cupy as cp
import numpy as np
from skimage.util import view_as_windows

from glcm_cupy.conf import *
from glcm_cupy.glcm_base import GLCMBase


def glcm_cross(
    im: np.ndarray,
    radius: int = 2,
    bin_from: int = 256,
    bin_to: int = 256,
    max_partition_size: int = MAX_PARTITION_SIZE,
    max_threads: int = MAX_THREADS,
    normalized_features: bool = True,
    verbose: bool = True,
    ix_combos: List[Tuple[int, int]] | None = None
) -> np.ndarray:
    """ Runs the Cross GLCM algorithm

    Notes:
        This will do a combinatorix of image channels to yield GLCMs

    Examples:
        To scale down the image from a 128 max value to 32, we use
        bin_from = 128, bin_to = 32.

        The range will collapse from 128 to 32.

        This thus optimizes the GLCM speed.

    Args:
        im: Image to Process
        radius: Radius of Window
        bin_from: Binarize from.
        bin_to: Binarize to.
        max_partition_size: Maximum number of windows to parse at once
        max_threads: Maximum threads for CUDA
        normalized_features: Whether to normalize features to [0, 1]
        verbose: Whether to enable TQDM logging
        ix_combos: Set of combinations to Cross with. If None, then all.
            E.g. ix_combos = [(0, 1), (0, 2), ..., (2, 3), (3, 3)]

    Returns:
        GLCM Features
    """
    return GLCMCross(
        radius=radius,
        bin_from=bin_from,
        bin_to=bin_to,
        max_partition_size=max_partition_size,
        max_threads=max_threads,
        normalized_features=normalized_features,
        ix_combos=ix_combos,
        verbose=verbose
    ).run(im)


@dataclass
class GLCMCross(GLCMBase):
    """
    Args:
        ix_combos: Set of combinations to Cross with. If None, then all.
            E.g. ix_combos = [(0, 1), (0, 2), ..., (2, 3), (3, 3)]
    """
    ix_combos: List[Tuple[int, int]] | None = None

    def ch_combos(self, im: np.ndarray) -> List[np.ndarray]:
        """ Get Image Channel Combinations """
        if self.ix_combos is None:
            # noinspection PyTypeChecker
            self.ix_combos = list(combinations(range(im.shape[-1]), 2))
        return [im[..., ix_combo] for ix_combo in self.ix_combos]

    def glcm_cells(self, im: np.ndarray) -> float:
        """ Total number of GLCM cells to process """
        shape = self.glcm_shape(im[..., 0])
        return np.prod(shape) * len(self.ch_combos(im))

    def glcm_shape(self, im_chn: np.ndarray) -> Tuple[int, int]:
        """ Get per-channel shape after GLCM """

        return im_chn.shape[0] - 2 * self.radius, \
               im_chn.shape[1] - 2 * self.radius

    def _from_im(self, im: np.ndarray) -> np.ndarray:
        """ Generates the GLCM from a multichannel image

        Args:
            im: A (H, W, C) image as ndarray

        Returns:
            The GLCM Array with shape (H, W, C, F)
        """

        ch_combos = self.ch_combos(im)
        if len(ch_combos) == 0:
            raise ValueError(f"Cross GLCM needs >= 2 channels to combine")
        glcm_chs = [self._from_channel(ch_combo) for ch_combo in ch_combos]

        return np.stack(glcm_chs, axis=2)

    def make_windows(self, im_chn: np.ndarray) -> List[Tuple[np.ndarray,
                                                             np.ndarray]]:
        """ Convert a image dual channel np.ndarray, to GLCM IJ windows.

        Examples:

            Input 4 x 4 image. Radius = 1
              1-2-+-+-+         1-+-+-+    2-+-+-+
             /3 4    /|         |     |    |     |    3-+-+-+    4-+-+-+
            1-2-+-+-+ |       1-+-+-+ |  2-+-+-+ |    |     |    |     |
            4 3     | |       |     | |  |     | |  3-+-+-+ |  4-+-+-+ |
            |       | | ----> |     | |  |     | |  |     | |  |     | |
            |       | +       |     |-+  |     |-+  |     | |  |     | |
            |       |/        +-+-+-+    +-+-+-+    |     |-+  |     |-+
            +-+-+-+-+                               +-+-+-+    +-+-+-+
            4 x 4 x 2                  2 x 2 x 3 x 3  ----> 4 x 9
                                       +---+   +---+  flat
                                       flat    flat

            The output will be flattened on the x, y,

        Notes:
            This is returned as a List to be similar to GLCM

        Args:
            im_chn: Input Image

        Returns:
            A 1 item List of IJ windows as a Tuple[I, J]
            Each with shape (Windows, Cells)
            [
                I: (Window Ix, Cell Ix), J: (Window Ix, Cell Ix)
            ]
        """

        if im_chn.ndim != 3:
            raise ValueError(
                f"Image must be 3 dimensional. (H, W, 2)"
                f"shape={im_chn.shape}"
            )

        glcm_h, glcm_w, *_ = self.glcm_shape(im_chn)
        if glcm_h <= 0 or glcm_w <= 0:
            raise ValueError(
                f"Step Size & Diameter exceeds size for windowing. "
                f"shape[0] {im_chn.shape[0]} "
                f"- 2 * step_size {self.step_size} "
                f"- 2 * radius {self.radius} <= 0 or"
                f"shape[1] {im_chn.shape[1]} "
                f"- 2 * step_size {self.step_size} "
                f"- 2 * radius {self.radius} + 1 <= 0 was not satisfied."
            )

        i = cp.asarray(
            view_as_windows(im_chn[..., 0], (self._diameter, self._diameter)))
        j = cp.asarray(
            view_as_windows(im_chn[..., 1], (self._diameter, self._diameter)))

        i = i.reshape((-1, *i.shape[-2:])) \
            .reshape((i.shape[0] * i.shape[1], -1))
        j = j.reshape((-1, *j.shape[-2:])) \
            .reshape((j.shape[0] * j.shape[1], -1))

        return [(i, j)]
