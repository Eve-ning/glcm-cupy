from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List

import cupy as cp
import numpy as np
from cupy import RawKernel
from tqdm import tqdm

from glcm_cupy.conf import *
from glcm_cupy.kernel import glcm_module


@dataclass
class GLCMBase:
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
        directions: Directions to pair the windows.
        max_partition_size: Maximum number of windows to parse at once
        max_threads: Maximum number of threads to use per block
        normalize_features: Whether to normalize features to [0, 1]
    """
    step_size: int = 1
    radius: int = 2
    bin_from: int = 256
    bin_to: int = 256
    max_partition_size: int = MAX_PARTITION_SIZE
    max_threads: int = MAX_THREADS
    normalized_features: bool = True

    glcm_create_kernel: RawKernel = field(
        default=glcm_module.get_function('glcmCreateKernel'),
        init=False
    )
    glcm_feature_kernel_0: RawKernel = field(
        default=glcm_module.get_function('glcmFeatureKernel0'),
        init=False
    )
    glcm_feature_kernel_1: RawKernel = field(
        default=glcm_module.get_function('glcmFeatureKernel1'),
        init=False
    )
    glcm_feature_kernel_2: RawKernel = field(
        default=glcm_module.get_function('glcmFeatureKernel2'),
        init=False
    )

    glcm: cp.ndarray = field(init=False)
    features: cp.ndarray = field(init=False)
    progress: tqdm = field(init=False)

    def __post_init__(self):

        self.glcm = cp.zeros(
            (self.max_partition_size, self.bin_to, self.bin_to),
            dtype=cp.float32
        )
        self.features = cp.zeros(
            (self.max_partition_size, NO_OF_FEATURES),
            dtype=cp.float32
        )
        if self.radius < 0:
            raise ValueError(f"Radius {self.radius} should be > 0)")

        if self.bin_to not in range(2, MAX_VALUE_SUPPORTED + 1):
            raise ValueError(
                f"Target Bins {self.bin_to} must be "
                f"[2, {MAX_VALUE_SUPPORTED}]. "
            )
        if self.step_size <= 0:
            raise ValueError(f"Step Size {step_size} should be >= 1")

    @property
    def _diameter(self):
        return self.radius * 2 + 1

    @abstractmethod
    def glcm_cells(self, im:np.ndarray) -> float:
        """ Total number of GLCM Cells"""
        ...

    def run(self, im: np.ndarray):
        """ Executes running GLCM. Returns the GLCM Feature array

        Args:
            im: 3D Image to process

        Returns:
            An np.ndarray of Shape,
             3D: (rows, cols, channels, features),

        """
        self.progress = tqdm(total=self.glcm_cells,
                             desc="GLCM Progress",
                             unit=" Cells",
                             unit_scale=True)

        im = self._binner(im, self.bin_from, self.bin_to)
        if im.ndim == 3:
            raise ValueError("Only 3D images allowed.")
        return self._from_3dimage(im)

    @abstractmethod
    def _from_3dimage(self, im: np.ndarray) -> np.ndarray:
        """ Generates the GLCM from a multi band image

        Args:
            im: A 3 dim image as an ndarray

        Returns:
            The GLCM Array 4dim with shape
                rows, cols, channel, feature
        """

        ...

    @abstractmethod
    def make_windows(self, im: np.ndarray) -> \
        List[Tuple[np.ndarray, np.ndarray]]:
        ...

    @abstractmethod
    def glcm_shape(self, im: np.ndarray) -> Tuple:
        ...

    def _from_2dimage(self, im: np.ndarray) -> np.ndarray:
        """ Generates the GLCM from a single band image

        Args:
            im: Image in np.ndarray.

        Returns:
            The GLCM Array 3dim with shape rows, cols, feature
        """

        if im.ndim != 2:
            raise ValueError(f"Image must be 2 dimensional. im.ndim={im.ndim}")

        glcm_h, glcm_w, *_ = self.glcm_shape(im)

        glcm_features = [
            self.glcm_window_ij(i, j)
                .reshape(glcm_h, glcm_w, NO_OF_FEATURES)
                .get() for i, j in self.make_windows(im)
        ]

        ar = np.stack(glcm_features).mean(axis=0)

        return self.normalize_features(ar) if self.normalized_features else ar

    def glcm_window_ij(self, windows_i: np.ndarray, windows_j: np.ndarray):
        windows_count = windows_i.shape[0]
        glcm_features = cp.zeros(
            (windows_count, NO_OF_FEATURES),
            dtype=cp.float32
        )

        # If 45000 windows, 10000 partition size,
        # We have 5 partitions (10K, 10K, 10K, 10K, 5K)
        partition_count = math.ceil(windows_count / self.max_partition_size)

        for partition in range(partition_count):
            with cp.cuda.Device() as dev:
                # As above, we have an uneven partition
                # Though [1,2,3][:5] == [1,2,3]
                # This means windows_part_i will be correctly partitioned.
                start = partition * self.max_partition_size
                end = (partition + 1) * self.max_partition_size
                part_i = windows_i[start:end]
                part_j = windows_j[start:end]

                # We don't need to figure out the leftover partition size
                # We can simply yield the length of the leftover partition
                windows_part_count = part_i.shape[0]

                glcm_features[start:start + windows_part_count] = \
                    self.glcm_ij(part_i, part_j)
                dev.synchronize()
                self.progress.update(windows_part_count)

        return glcm_features

    def glcm_ij(self,
                i: np.ndarray,
                j: np.ndarray):
        """ Generate the GLCM from the I J Window

        Examples:

            >>> ar_0 = np.random.randint(0, 100, 10, dtype=np.uint8)
            >>> ar_1 = np.random.randint(0, 100, 10, dtype=np.uint8)
            >>> g = GLCM().glcm_ij(ar_0[...,np.newaxis], ar_1[...,np.newaxis])

        Notes:
            i must be the same shape as j
            i.shape, j.shape should be [partition_size, window_size]

            For example, if you want to process 100 windows at once,
            each window with 25 cells, you should have
            i.shape == j.shape == (100, 25)

        Args:
            i: 1st np.ndarray
            j: 2nd np.ndarray

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

        # Reset data
        self.glcm[:] = 0
        self.features[:] = 0

        no_of_windows = i.shape[0]
        no_of_values = self._diameter ** 2

        if i.dtype != np.uint8 or j.dtype != np.uint8:
            raise ValueError(
                f"Image dtype must be np.uint8,"
                f" i: {i.dtype} j: {j.dtype}"
            )

        grid = self._calc_grid_size(no_of_windows,
                                    self.bin_to,
                                    self.max_threads)
        self.glcm_create_kernel(
            grid=grid,
            block=(self.max_threads,),
            args=(
                i,
                j,
                self.bin_to,
                no_of_values,
                no_of_windows,
                self.glcm,
                self.features
            )
        )

        feature_args = dict(
            grid=grid, block=(self.max_threads,),
            args=(self.glcm,
                  self.bin_to,
                  no_of_values,
                  no_of_windows,
                  self.features)
        )
        self.glcm_feature_kernel_0(**feature_args)
        self.glcm_feature_kernel_1(**feature_args)
        self.glcm_feature_kernel_2(**feature_args)

        return self.features[:no_of_windows]

    def normalize_features(self, ar: np.ndarray):
        """ This scales the glcm features to [0, 1] """
        ar[..., CONTRAST] /= (self.bin_to - 1) ** 2
        ar[..., MEAN] /= (self.bin_to - 1)
        ar[..., VAR] /= (self.bin_to - 1) ** 2
        ar[..., CORRELATION] += 1
        ar[..., CORRELATION] /= 2
        return ar

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

        # Blocks to support features
        blocks_req_glcm_features = \
            window_count * glcm_size * glcm_size / thread_per_block

        # Blocks to support glcm populating
        blocks_req_glcm_populate = glcm_size * window_count

        # Take the maximum
        blocks_req = max(blocks_req_glcm_features, blocks_req_glcm_populate)

        # We split it to 2 dims: A -> B x B
        b = int(blocks_req ** 0.5) + 1
        return b, b

    @staticmethod
    def _binner(im: np.ndarray, bin_from: int, bin_to: int) -> np.ndarray:
        """ Bins an image from a certain bin to another

        Args:
            im: Image as np.ndarray
            bin_from: From the Bin of input image
            bin_to: To the Bin of output image

        Returns:
            Binned Image

        """
        return (im.astype(np.float32) / bin_from * bin_to).astype(np.uint8)
