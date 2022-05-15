import itertools
from dataclasses import dataclass

import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm

from glcm_cupy.conf import NO_OF_FEATURES
from glcm_cupy.glcm_py_base import GLCMPyBase
from glcm_cupy.utils import normalize_features


def glcm_cross_py_im(im: np.ndarray, bin_from: int, bin_to: int,
                     radius: int = 2):
    return GLCMCrossPy(bin_from=bin_from,
                       bin_to=bin_to,
                       radius=radius).glcm_im(im)


def glcm_cross_py_chn(im_chn: np.ndarray,
                      bin_from: int,
                      bin_to: int,
                      radius: int = 2):
    return GLCMCrossPy(bin_from=bin_from,
                       bin_to=bin_to,
                       radius=radius).glcm_chn(im_chn)


def glcm_cross_py_ij(i: np.ndarray,
                     j: np.ndarray,
                     bin_from: int, bin_to: int):
    return GLCMCrossPy(bin_from=bin_from,
                       bin_to=bin_to).glcm_ij(i, j)


@dataclass
class GLCMCrossPy(GLCMPyBase):

    def glcm_chn(self, ar: np.ndarray):
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

        return normalize_features(feature_ar, self.bin_to)

    def glcm_im(self, ar: np.ndarray):
        combos = list(itertools.combinations(range(ar.shape[-1]), 2))
        return np.stack([self.glcm_chn(ar[..., combo]) for combo in combos],
                        axis=2)
