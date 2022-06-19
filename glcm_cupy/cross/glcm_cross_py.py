import itertools
from dataclasses import dataclass
from typing import List, Tuple

import cupy as cp
import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm

from glcm_cupy.conf import NO_OF_FEATURES, ndarray
from glcm_cupy.glcm_py_base import GLCMPyBase
from glcm_cupy.utils import normalize_features


def glcm_cross_py_im(im: np.ndarray, bin_from: int, bin_to: int,
                     radius: int = 2, ix_combos: List[Tuple[int, int]] = None):
    return GLCMCrossPy(bin_from=bin_from,
                       bin_to=bin_to,
                       radius=radius,
                       ix_combos=ix_combos).glcm_im(im)


@dataclass
class GLCMCrossPy(GLCMPyBase):
    ix_combos: List[Tuple[int, int]] = None

    def glcm_chn(self, ar: ndarray):
        # Convert to compatible types
        if isinstance(ar, cp.ndarray):
            ar = (ar / self.bin_from * self.bin_to).astype(cp.uint8)
        else:
            ar = (ar / self.bin_from * self.bin_to).astype(np.uint8)

        def flat(ar_: ndarray):
            ar_ = ar_.reshape((-1, self.diameter, self.diameter))
            return ar_.reshape((ar_.shape[0], -1))

        ar_w_i = flat(
            view_as_windows(ar[..., 0], (self.diameter, self.diameter))
        )
        ar_w_j = flat(
            view_as_windows(ar[..., 1], (self.diameter, self.diameter))
        )

        if isinstance(ar, cp.ndarray):
            feature_ar = cp.zeros((ar_w_i.shape[0], NO_OF_FEATURES))
        else:
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

    def glcm_im(self, ar: ndarray):
        if self.ix_combos is None:
            self.ix_combos = list(
                itertools.combinations(range(ar.shape[-1]), 2))

        if isinstance(ar, cp.ndarray):
            return cp.stack(
                [self.glcm_chn(ar[..., ix_combo]) for ix_combo in
                 self.ix_combos],
                axis=2
            )

        return np.stack(
            [self.glcm_chn(ar[..., ix_combo]) for ix_combo in self.ix_combos],
            axis=2
        )
