from dataclasses import dataclass
from typing import List

import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm

from glcm_cupy import *


def glcm_py_3d(ar: np.ndarray, bin_from: int, bin_to: int,
               radius: int = 2,
               step: int = 1):
    return GLCMPy(bin_from=bin_from,
                  bin_to=bin_to,
                  radius=radius,
                  step=step).glcm_3d(ar)


def glcm_py_2d(ar: np.ndarray,
               bin_from: int,
               bin_to: int,
               radius: int = 2,
               step: int = 1):
    return GLCMPy(bin_from=bin_from,
                  bin_to=bin_to,
                  radius=radius,
                  step=step).glcm_2d(ar)


def glcm_py_ij(i: np.ndarray,
               j: np.ndarray,
               bin_from: int, bin_to: int):
    return GLCMPy(bin_from=bin_from,
                  bin_to=bin_to).glcm_ij(i, j)


@dataclass
class GLCMPy:
    bin_from: int
    bin_to: int
    radius: int = 2
    step: int = 1

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
                homogeneity, contrast, asm, mean_i, mean_j,
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

        homogeneity = contrast = asm = mean_i = mean_j = var_i = var_j \
            = correlation = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
                contrast += glcm[i, j] * (i - j) ** 2
                asm += glcm[i, j] ** 2
                mean_i += glcm[i, j] * i
                mean_j += glcm[i, j] * j

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                var_i += glcm[i, j] * (i - mean_i) ** 2
                var_j += glcm[i, j] * (j - mean_j) ** 2

        # Variances cannot be 0 else ZeroDivisionError
        if var_i != 0 and var_j != 0:
            for i in range(glcm.shape[0]):
                for j in range(glcm.shape[1]):
                    correlation += \
                        glcm[i, j] \
                        * (i - mean_i) \
                        * (j - mean_j) \
                        / ((var_i * var_j) ** 0.5)

        return [homogeneity, contrast, asm, mean_i, mean_j, var_i, var_j,
                correlation]

    def glcm_2d(self, ar: np.ndarray):
        ar = (ar / self.bin_from * self.bin_to).astype(np.uint8)
        ar_w = view_as_windows(ar, (self.diameter, self.diameter))

        def flat(ar: np.ndarray):
            ar = ar.reshape((-1, self.diameter, self.diameter))
            return ar.reshape((ar.shape[0], -1))

        ar_w_i = flat(ar_w[self.step:-self.step, self.step:-self.step])
        ar_w_j_sw = flat(ar_w[self.step * 2:, :-self.step * 2])
        ar_w_j_s = flat(ar_w[self.step * 2:, self.step:-self.step])
        ar_w_j_se = flat(ar_w[self.step * 2:, self.step * 2:])
        ar_w_j_e = flat(ar_w[self.step:-self.step, self.step * 2:])

        feature_ar = np.zeros((ar_w_i.shape[0], 4, 8))

        for j_e, ar_w_j in enumerate(
            (ar_w_j_sw, ar_w_j_s, ar_w_j_se, ar_w_j_e)):
            for e, (i, j) in tqdm(enumerate(zip(ar_w_i, ar_w_j)),
                                  total=ar_w_i.shape[0]):
                feature_ar[e, j_e] = self.glcm_ij(i, j)

        h, w = ar_w.shape[:2]
        feature_ar = feature_ar.mean(axis=1)

        feature_ar = feature_ar.reshape(
            (h - self.step * 2, w - self.step * 2, 8))
        feature_ar[..., CONTRAST] /= (self.bin_to - 1) ** 2
        feature_ar[..., MEAN_I] /= (self.bin_to - 1)
        feature_ar[..., MEAN_J] /= (self.bin_to - 1)
        feature_ar[..., VAR_I] /= (self.bin_to - 1) ** 2
        feature_ar[..., VAR_J] /= (self.bin_to - 1) ** 2
        feature_ar[..., CORRELATION] += 1
        feature_ar[..., CORRELATION] /= 2

        return feature_ar

    def glcm_3d(self, ar: np.ndarray):
        return np.stack([self.glcm_2d(ch) for ch in ar])
