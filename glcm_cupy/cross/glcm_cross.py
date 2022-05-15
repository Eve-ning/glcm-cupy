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
        radius: Radius of Window
        bin_from: Binarize from.
        bin_to: Binarize to.
        max_partition_size: Maximum number of windows to parse at once
        max_threads: Maximum threads for CUDA
        normalize_features: Whether to normalize features to [0, 1]

    Returns:
        GLCM Features
    """
    return GLCMCross(radius, bin_from, bin_to,
                     max_partition_size, max_threads,
                     normalize_features).run(im)


@dataclass
class GLCMCross(GLCMBase):

    @staticmethod
    def ch_combos(im: np.ndarray) -> List[np.ndarray]:
        return [im[..., combo]
                for combo
                in combinations(range(im.shape[-1]), 2)]

    def glcm_cells(self, im: np.ndarray) -> float:
        shape = self.glcm_shape(im[..., 0])
        return np.prod(shape) * len(self.ch_combos(im))

    def glcm_shape(self, im_chn: np.ndarray) -> Tuple[int, int]:
        """ Get per-channel shape after GLCM """

        return im_chn.shape[0] - 2 * self.radius, \
               im_chn.shape[1] - 2 * self.radius

    def _from_im(self, im: np.ndarray) -> np.ndarray:
        """ Generates the GLCM from a multi band image

        Args:
            im: A 3 dim image as an ndarray

        Returns:
            The GLCM Array 4dim with shape
                rows, cols, channel, feature
        """

        ch_combos = self.ch_combos(im)
        if len(ch_combos) == 0:
            raise ValueError(f"Cross GLCM needs >= 2 channels to combine")
        glcm_chs = [self._from_channel(ch_combo) for ch_combo in ch_combos]

        return np.stack(glcm_chs, axis=2)

    def make_windows(self, im_chn: np.ndarray) -> List[Tuple[np.ndarray,
                                                             np.ndarray]]:
        """ Convert 3D image np.ndarray with 2 channels to IJ windows.

        Args:
            im_chn: Input Image. Must be of shape (h, w, 2)

        Returns:
            A List of I, J windows based on the directions.

            For each window:
                The first dimension: xy flat window indexes,
                the last dimension: xy flat indexes within each window.

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
