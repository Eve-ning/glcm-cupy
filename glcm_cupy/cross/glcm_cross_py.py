import itertools
from dataclasses import dataclass
from typing import List

import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm

from glcm_cupy import *
from glcm_cupy.conf import NO_OF_FEATURES


def glcm_cross_py_3d(ar: np.ndarray, bin_from: int, bin_to: int,
                     radius: int = 2):
    return GLCMCrossPy(bin_from=bin_from,
                       bin_to=bin_to,
                       radius=radius).glcm_3d(ar)


def glcm_cross_py_2d(ar: np.ndarray,
                     bin_from: int,
                     bin_to: int,
                     radius: int = 2):
    return GLCMCrossPy(bin_from=bin_from,
                       bin_to=bin_to,
                       radius=radius).glcm_2d(ar)


def glcm_cross_py_ij(i: np.ndarray,
                     j: np.ndarray,
                     bin_from: int, bin_to: int):
    return GLCMCrossPy(bin_from=bin_from,
                       bin_to=bin_to).glcm_ij(i, j)


@dataclass
class GLCMCrossPy:
    bin_from: int
    bin_to: int
    radius: int = 2

    @property
    def diameter(self) -> int:
        return self.radius * 2 + 1

    def glcm_ij(self,
                i: np.ndarray,
                j: np.ndarray) -> List[float]:
        """ Calculate the expected GLCM features using Python

        Notes:
            This is to assert the actual values in the tests.
            Technically can be used in production though very slow.

        Args:
            i: Window I, may be ndim
            j: Window J, may be ndim

        Returns:
            List of these features [
                homogeneity, contrast, asm, mean, mean_j,
                var_i, var_j, correlation
            ].
            Values as float.
        """

        i_flat = i.flatten()
        j_flat = j.flatten()
        assert len(i_flat) == len(j_flat), \
            f"The shapes for i {i.shape} != j {j.shape}."

        glcm = np.zeros((self.bin_to, self.bin_to), dtype=float)

        # Populate the GLCM
        for i_, j_ in zip(i_flat, j_flat):
            glcm[i_, j_] += 1
            glcm[j_, i_] += 1

        # Convert to probability
        glcm /= len(i_flat) * 2

        homogeneity = contrast = asm = mean = var = correlation = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
                contrast += glcm[i, j] * (i - j) ** 2
                asm += glcm[i, j] ** 2
                mean += glcm[i, j] * i

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                var += glcm[i, j] * (i - mean) ** 2

        # Variances cannot be 0 else ZeroDivisionError
        if var != 0:
            for i in range(glcm.shape[0]):
                for j in range(glcm.shape[1]):
                    correlation += glcm[i, j] * (i - mean) * (j - mean) / var

        return [homogeneity, contrast, asm, mean, var, correlation]

    def glcm_2d(self, ar: np.ndarray):
        ar = (ar / self.bin_from * self.bin_to).astype(np.uint8)

        def flat(ar_: np.ndarray):
            ar_ = ar_.reshape((-1, self.diameter, self.diameter))
            return ar_.reshape((ar_.shape[0], -1))

        ar_w_i = flat(
            view_as_windows(ar[..., 0], (self.diameter, self.diameter))
        )
        ar_w_j = flat(
            view_as_windows(ar[..., 1], (self.diameter, self.diameter))
        )

        feature_ar = np.zeros((ar_w_i.shape[0], NO_OF_FEATURES))
        for e, (i, j) in tqdm(enumerate(zip(ar_w_i, ar_w_j)),
                              total=ar_w_i.shape[0]):
            feature_ar[e] = self.glcm_ij(i, j)

        feature_ar = feature_ar.reshape(
            (ar.shape[0] - self.radius * 2,
             ar.shape[1] - self.radius * 2,
             NO_OF_FEATURES)
        )
        feature_ar[..., CONTRAST] /= (self.bin_to - 1) ** 2
        feature_ar[..., MEAN] /= (self.bin_to - 1)
        feature_ar[..., VAR] /= (self.bin_to - 1) ** 2
        feature_ar[..., CORRELATION] += 1
        feature_ar[..., CORRELATION] /= 2

        return feature_ar

    def glcm_3d(self, ar: np.ndarray):
        combos = list(itertools.combinations(range(ar.shape[-1]), 2))
        return np.stack([self.glcm_2d(ar[..., combo]) for combo in combos])
