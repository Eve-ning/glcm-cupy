from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cupy as cp
import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm

from kernel import glcm_module

MAX_VALUE_SUPPORTED = 255
NO_OF_VALUES_SUPPORTED = 256 ** 2
MAX_RADIUS_SUPPORTED = 127

MAX_THREADS = 512  # Lowest Maximum supported threads.

NO_OF_FEATURES = 8

PARTITION_SIZE = 10000


@dataclass
class GLCM:
    """
    
    Args:
        max_value: Maximum value of the image, default 256
        step_size: Step size of the window
        radius: Radius of the windows
        bins: Bin reduction. If None, then no reduction is done

    """

    max_value: int = MAX_VALUE_SUPPORTED
    step_size: int = 1
    radius: int = 2
    bins: int = 256

    threads = MAX_VALUE_SUPPORTED + 1

    HOMOGENEITY = 0
    CONTRAST = 1
    ASM = 2
    MEAN_I = 3
    MEAN_J = 4
    VAR_I = 5
    VAR_J = 6
    CORRELATION = 7

    @property
    def diameter(self):
        return self.radius * 2 + 1

    @property
    def no_of_values(self):
        return self.diameter ** 2

    def __post_init__(self):
        self.i_flat = cp.zeros((self.diameter ** 2,), dtype=cp.uint8)
        self.j_flat = cp.zeros((self.diameter ** 2,), dtype=cp.uint8)
        self.glcm = cp.zeros((self.max_value + 1) ** 2, dtype=cp.uint8)
        self.features = cp.zeros(8, dtype=cp.float32)

        if not 1 <= self.max_value <= MAX_VALUE_SUPPORTED:
            raise ValueError(
                f"Max value {self.max_value} should be in [1, {MAX_VALUE_SUPPORTED}]")
        if not 0 <= self.radius <= MAX_RADIUS_SUPPORTED:
            f"Radius {self.radius} should be in [0, {MAX_RADIUS_SUPPORTED}]"
        if not (2 <= self.bins <= MAX_VALUE_SUPPORTED + 1):
            raise ValueError(
                f"Bins {self.bins} should be in [2, {MAX_VALUE_SUPPORTED + 1}]. "
                f"If bins == 256, just use None."
            )
        if not 1 <= self.step_size:
            raise ValueError(
                f"Step Size {self.step_size} should be >= 1"
                f"If bins == 256, just use None."
            )

        os.environ['CUPY_EXPERIMENTAL_SLICE_COPY'] = '1'

    @staticmethod
    def binarize(im: np.ndarray, from_bins: int, to_bins: int):
        """ Binarize an image from a certain bin to another

        Args:
            im: Image as np.ndarray
            from_bins: From the Bin of input image
            to_bins: To the Bin of output image

        Returns:
            Binarized Image

        """
        return (im / from_bins * to_bins).astype(np.uint8)

    def from_3dimage(self,
                     im: np.ndarray):
        """ Generates the GLCM from a multi band image

        Args:
            im: A 3 dim image as an ndarray

        Returns:
            The GLCM Array 4dim with shape
                rows, cols, channel, feature
        """

        glcm_chs = []
        for ch in range(im.shape[-1]):
            glcm_chs.append(self.from_2dimage(im[..., ch]))

        return np.stack(glcm_chs, axis=2)

    def from_2dimage(self,
                     im: np.ndarray) -> np.ndarray:
        """ Generates the GLCM from a single band image

        Notes:
            This will actively partition the processing by blocks
            of PARTITION_SIZE.
            This allows for a reduction in GLCM creation.

        Args:
            im: Image in np.ndarray. Cannot be in cp.ndarray

        Returns:
            The GLCM Array 3dim with shape
                rows, cols, feature
        """

        # This will yield a shape (window_i, window_j, row, col)
        # E.g. 100x100 with 5x5 window -> 96, 96, 5, 5
        windows_ij = view_as_windows(im, (self.diameter, self.diameter))

        # We flatten the cells as cell order is not important
        windows_ij = windows_ij.reshape((*windows_ij.shape[:-2], -1))

        # Yield Windows and flatten the windows
        windows_i = windows_ij[:-self.step_size, :-self.step_size] \
            .reshape((-1, windows_ij.shape[-1]))
        windows_j = windows_ij[self.step_size:, self.step_size:] \
            .reshape((-1, windows_ij.shape[-1]))

        glcm_features = cp.zeros((windows_i.shape[0], NO_OF_FEATURES),
                                 dtype=cp.float32)

        windows_count = windows_i.shape[0]
        glcm_ix = 0
        for partition in range(math.ceil(windows_count / PARTITION_SIZE)):
            for i, j in tqdm(
                zip(windows_i[
                    (start := partition * PARTITION_SIZE):
                    (end := (partition + 1) * PARTITION_SIZE)
                    ],
                    windows_j[start:end]
                    ),
                total=len(windows_i[start:end])):
                glcm_features[glcm_ix] = self._from_windows(i, j)
                glcm_ix += 1

        return glcm_features.reshape(windows_ij.shape[0] - self.step_size,
                                     windows_ij.shape[1] - self.step_size,
                                     NO_OF_FEATURES)

    def _from_windows(self,
                      i: np.ndarray,
                      j: np.ndarray, ) -> np.ndarray:
        """ Generate the GLCM from the I J Window

        Notes:
            i must be the same shape as j

        Args:
            i: I Window
            j: J Window

        Returns:
            The GLCM array, of size (8,)

        """

        assert i.shape == j.shape, f"Shape of i {i.shape} != j {j.shape}"

        i = self.binarize(i, self.max_value, self.bins)
        j = self.binarize(j, self.max_value, self.bins)

        if i.dtype != np.uint8 or j.dtype != np.uint8:
            raise ValueError(
                f"Image dtype must be np.uint8, i: {i.dtype} j: {j.dtype}"
            )

        blocks = int(max(i.max(), j.max())) + 1
        self.i_flat = cp.asarray(i.flatten())
        self.j_flat = cp.asarray(j.flatten())
        self.glcm[:] = 0
        self.features[:] = 0

        glcm_k0 = glcm_module.get_function('glcm_0')
        glcm_k1 = glcm_module.get_function('glcm_1')
        glcm_k2 = glcm_module.get_function('glcm_2')
        glcm_k0(
            grid=(blocks,),
            block=(self.threads,),
            args=(
                self.i_flat, self.j_flat,
                self.max_value, self.no_of_values,
                self.glcm, self.features
            )
        )
        glcm_k1(
            grid=(blocks,),
            block=(self.threads,),
            args=(
                self.glcm, self.max_value,
                self.no_of_values, self.features
            )
        )
        glcm_k2(
            grid=(blocks,),
            block=(self.threads,),
            args=(
                self.glcm, self.max_value,
                self.no_of_values, self.features
            )
        )

        return self.features
