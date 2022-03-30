from __future__ import annotations

import math
from typing import Tuple

import cupy as cp
import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm

from glcm_pycuda.conf import *
from glcm_pycuda.kernel import glcm_module


class GLCM:

    def __init__(self, step_size: int = 1, radius: int = 2,
                 bin_from: int = 256,
                 bin_to: int = 256):
        """ Initialize Settings for GLCM

        Examples:
            To scale down the image from a 128 max value to 32, we use
            bin_from = 128, bin_to = 32.

            The range will collapse from 128 to 32.

            This thus optimizes the GLCM speed.

        Args:
            step_size: Stride Between GLCMs
            radius: Radius of Window
            bin_from: Binarize from.
            bin_to: Binarize to.
        """
        self.step_size = step_size
        self.radius = radius
        self.bin_from = bin_from
        self.bin_to = bin_to

        self.i_gpu = cp.zeros((self.diameter ** 2,), dtype=cp.uint8)
        self.j_gpu = cp.zeros((self.diameter ** 2,), dtype=cp.uint8)

        if self.radius < 0:
            raise ValueError(
                f"Radius {radius} should be > 0)"
            )
        if not bin_to in range(2, MAX_VALUE_SUPPORTED + 1):
            raise ValueError(
                f"Target Bins {bin_to} should be "
                f"[2, {MAX_VALUE_SUPPORTED}]. "
            )
        if self.step_size <= 0:
            raise ValueError(
                f"Step Size {step_size} should be >= 1"
            )

        self.glcm_create_kernel = \
            glcm_module.get_function('glcmCreateKernel')
        self.glcm_feature_kernel_0 = \
            glcm_module.get_function('glcmFeatureKernel0')
        self.glcm_feature_kernel_1 = \
            glcm_module.get_function('glcmFeatureKernel1')
        self.glcm_feature_kernel_2 = \
            glcm_module.get_function('glcmFeatureKernel2')

    @property
    def diameter(self):
        return self.radius * 2 + 1

    def from_3dimage(self, im: np.ndarray) -> np.ndarray:
        """ Generates the GLCM from a multi band image

        Args:
            im: A 3 dim image as an ndarray

        Returns:
            The GLCM Array 4dim with shape
                rows, cols, channel, feature
        """

        glcm_chs = []
        for ch in range(im.shape[-1]):  # Channel Loop
            glcm_chs.append(self.from_2dimage(im[..., ch]))

        return np.stack(glcm_chs, axis=2)

    def from_2dimage(self,
                     im: np.ndarray) -> np.ndarray:
        """ Generates the GLCM from a single band image

        Notes:
            Partitions the image into blocks of PARTITION_SIZE,
            reducing allocated GLCM Memory.

        Args:
            im: Image in np.ndarray.

        Returns:
            The GLCM Array 3dim with shape
                rows, cols, feature
        """

        if im.ndim != 2:
            raise ValueError(
                f"Image must be 2 dimensional. im.ndim={im.ndim}"
            )

        # Both dims are xy flattened.
        # windows.shape == [window_ix, cell_ix]
        windows_i, windows_j = \
            self._make_windows(im, self.diameter, self.step_size)

        windows_count = windows_i.shape[0]

        glcm_features = cp.zeros(
            (windows_count, NO_OF_FEATURES),
            dtype=cp.float32
        )

        # If 45000 windows, 10000 partition size,
        # We have 5 partitions (10K, 10K, 10K, 10K, 5K)
        partition_count = math.ceil(windows_count / MAX_PARTITION_SIZE)

        for partition in tqdm(range(partition_count)):
            with cp.cuda.Device() as dev:
                # As above, we have an uneven partition
                # Though [1,2,3][:5] == [1,2,3]
                # This means windows_part_i will be correctly partitioned.
                windows_part_i = windows_i[
                                 (start := partition * MAX_PARTITION_SIZE):
                                 (end := (partition + 1) * MAX_PARTITION_SIZE)
                                 ]
                windows_part_j = windows_j[start:end]

                # We don't need to figure out the leftover partition size
                # We can simply yield the length of the leftover partition
                windows_part_count = windows_part_i.shape[0]

                glcm_features[start:start + windows_part_count] = \
                    self._from_windows(
                        windows_part_i,
                        windows_part_j
                    )
                dev.synchronize()

        return glcm_features.reshape(
            im.shape[0] - self.radius * 2 - self.step_size,
            im.shape[1] - self.radius * 2 - self.step_size,
            NO_OF_FEATURES
        )

    def _from_windows(self,
                      i: np.ndarray,
                      j: np.ndarray):
        """ Generate the GLCM from the I J Window

        Examples:

            >>> ar_0 = np.random.randint(0, 100, 10, dtype=np.uint8)
            >>> ar_1 = np.random.randint(0, 100, 10, dtype=np.uint8)
            >>> g = GLCM()._from_windows(ar_0[...,np.newaxis], ar_1[...,np.newaxis])

        Notes:
            i must be the same shape as j
            i.shape, j.shape should be [partition_size, window_size]

            For example, if you want to process 100 windows at once,
            each window with 25 cells, you should have
            i.shape == j.shape == (100, 25)

        Args:
            i: I Window
            j: J Window

        Returns:
            The GLCM feature array, of size [partition_size, 8]

        """
        if i.ndim != 2 or j.ndim != 2:
            raise ValueError(
                f"The input dimensions must be 2. "
                f"i.ndim=={i.ndim}, j.ndim=={j.ndim}. "
                "The 1st Dim is the partitioned windows flattened, "
                "The 2nd is the window cells flattened"
            )

        if i.shape != j.shape:
            raise ValueError(f"Shape of i {i.shape} != j {j.shape}")

        # partition_size != MAX_PARTITION_SIZE
        # It may be the leftover partition.
        partition_size = i.shape[0]

        self.glcm = cp.zeros((partition_size, self.bin_to, self.bin_to),
                             dtype=cp.uint8)
        self.features = cp.zeros((partition_size, NO_OF_FEATURES),
                                 dtype=cp.float32)

        i = self._binarize(i, self.bin_from, self.bin_to)
        j = self._binarize(j, self.bin_from, self.bin_to)

        no_of_windows = i.shape[0]
        no_of_values = self.diameter ** 2

        if i.dtype != np.uint8 or j.dtype != np.uint8:
            raise ValueError(
                f"Image dtype must be np.uint8,"
                f" i: {i.dtype} j: {j.dtype}"
            )

        self.i_gpu = cp.asarray(i)
        self.j_gpu = cp.asarray(j)

        self.glcm_create_kernel(
            grid=(grid := self._calc_grid_size(no_of_windows, self.bin_to)),
            block=(MAX_THREADS,),
            args=(
                self.i_gpu,
                self.j_gpu,
                self.bin_to,
                no_of_values,
                no_of_windows,
                self.glcm,
                self.features
            )
        )

        feature_args = dict(
            grid=grid, block=(MAX_THREADS,),
            args=(self.glcm,
                  self.bin_to,
                  no_of_values,
                  no_of_windows,
                  self.features)
        )

        self.glcm_feature_kernel_0(**feature_args)
        self.glcm_feature_kernel_1(**feature_args)
        self.glcm_feature_kernel_2(**feature_args)
        del self.glcm
        return self.features[:no_of_windows]

    @staticmethod
    def _calc_grid_size(
        window_count: int,
        glcm_size: int,
        thread_per_block: int = MAX_THREADS
    ) -> Tuple[int, int]:
        """ Calculates the required grid size

        Notes:
            There's 2 points where the number of threads

        Returns:
            The optimal minimum grid shape for GLCM

        """

        # TODO: Do we need to div by thread_per_block?
        # Blocks to support features
        blocks_req_glcm_features = \
            window_count * glcm_size * glcm_size / thread_per_block

        # Blocks to support glcm populating
        blocks_req_glcm_populate = glcm_size * window_count

        # Take the maximum
        blocks_req = max(blocks_req_glcm_features, blocks_req_glcm_populate)

        # We split it to 2 dims: A -> B x B
        return (_ := int(blocks_req ** 0.5) + 1), _

    @staticmethod
    def _make_windows(im: np.ndarray,
                      diameter: int,
                      step_size: int) -> Tuple[np.ndarray, np.ndarray]:
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
            diameter: Diameter of Window
            step_size: Step Size between ij pairs

        Returns:
            The windows I, J suitable for GLCM.
            The first dimension: xy flat window indexes,
            the last dimension: xy flat indexes within each window.

        """

        # This will yield a shape (window_i, window_j, row, col)
        # E.g. 100x100 with 5x5 window -> 96, 96, 5, 5
        if im.shape[0] - step_size - diameter + 1 <= 0 or \
            im.shape[1] - step_size - diameter + 1 <= 0:
            raise ValueError(
                f"Step Size & Diameter exceeds size for windowing. "
                f"im.shape[0] {im.shape[0]} "
                f"- step_size {step_size} "
                f"- diameter{diameter} + 1 <= 0 or"
                f"im.shape[1] {im.shape[1]} "
                f"- step_size {step_size} "
                f"- diameter{diameter} + 1 <= 0 was not satisfied."
            )

        ij = view_as_windows(im, (diameter, diameter))
        i: np.ndarray = ij[:-step_size, :-step_size]
        j: np.ndarray = ij[step_size:, step_size:]

        i = i.reshape((-1, i.shape[-2], i.shape[-1])) \
            .reshape((i.shape[0] * i.shape[1], -1))
        j = j.reshape((-1, j.shape[-2], j.shape[-1])) \
            .reshape((j.shape[0] * j.shape[1], -1))

        return i, j

    @staticmethod
    def _binarize(im: np.ndarray, from_bins: int, to_bins: int) -> np.ndarray:
        """ Binarize an image from a certain bin to another

        Args:
            im: Image as np.ndarray
            from_bins: From the Bin of input image
            to_bins: To the Bin of output image

        Returns:
            Binarized Image

        """
        return (im.astype(np.float32) / from_bins * to_bins).astype(np.uint8)
