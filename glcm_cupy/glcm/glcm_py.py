from dataclasses import dataclass

from typing import Union

import cupy as cp
import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm

from glcm_cupy.conf import NO_OF_FEATURES
from glcm_cupy.glcm_py_base import GLCMPyBase
from glcm_cupy.utils import normalize_features


def glcm_py_im(ar: Union[np.ndarray, cp.ndarray], bin_from: int, bin_to: int,
               radius: int = 2,
               step: int = 1):
    return GLCMPy(bin_from=bin_from,
                  bin_to=bin_to,
                  radius=radius,
                  step=step).glcm_im(ar)


def glcm_py_chn(ar: Union[np.ndarray, cp.ndarray],
                bin_from: int,
                bin_to: int,
                radius: int = 2,
                step: int = 1):
    return GLCMPy(bin_from=bin_from,
                  bin_to=bin_to,
                  radius=radius,
                  step=step).glcm_chn(ar)


def glcm_py_ij(i: Union[np.ndarray, cp.ndarray],
               j: Union[np.ndarray, cp.ndarray],
               bin_from: int, bin_to: int):
    return GLCMPy(bin_from=bin_from,
                  bin_to=bin_to).glcm_ij(i, j)


@dataclass
class GLCMPy(GLCMPyBase):
    step: int = 1

    def glcm_chn(self, ar: Union[np.ndarray, cp.ndarray]):

        if isinstance(ar, cp.ndarray):
            ar = (ar / self.bin_from * self.bin_to).astype(cp.uint8)
        else:
            ar = (ar / self.bin_from * self.bin_to).astype(np.uint8)
        ar_w = view_as_windows(ar, (self.diameter, self.diameter))

        def flat(ar: Union[np.ndarray, cp.ndarray]):
            ar = ar.reshape((-1, self.diameter, self.diameter))
            return ar.reshape((ar.shape[0], -1))

        ar_w_i = flat(ar_w[self.step:-self.step, self.step:-self.step])
        ar_w_j_sw = flat(ar_w[self.step * 2:, :-self.step * 2])
        ar_w_j_s = flat(ar_w[self.step * 2:, self.step:-self.step])
        ar_w_j_se = flat(ar_w[self.step * 2:, self.step * 2:])
        ar_w_j_e = flat(ar_w[self.step:-self.step, self.step * 2:])

        if isinstance(ar, cp.ndarray):
            feature_ar = cp.zeros((ar_w_i.shape[0], 4, NO_OF_FEATURES))
        else:
            feature_ar = np.zeros((ar_w_i.shape[0], 4, NO_OF_FEATURES))

        for j_e, ar_w_j in enumerate(
            (ar_w_j_sw, ar_w_j_s, ar_w_j_se, ar_w_j_e)):
            for e, (i, j) in tqdm(enumerate(zip(ar_w_i, ar_w_j)),
                                  total=ar_w_i.shape[0]):
                feature_ar[e, j_e] = self.glcm_ij(i, j)

        h, w = ar_w.shape[:2]
        feature_ar = feature_ar.mean(axis=1)

        feature_ar = feature_ar.reshape(
            (h - self.step * 2, w - self.step * 2, NO_OF_FEATURES))

        return normalize_features(feature_ar, self.bin_to)

    def glcm_im(self, ar: Union[np.ndarray, cp.ndarray]):
        if isinstance(ar, cp.ndarray):
            return cp.stack([self.glcm_chn(ar[..., ch])
                             for ch
                             in range(ar.shape[-1])], axis=2)
        return np.stack([self.glcm_chn(ar[..., ch])
                         for ch
                         in range(ar.shape[-1])], axis=2)
