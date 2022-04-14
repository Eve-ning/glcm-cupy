from typing import Tuple

import cupy as cp
import numpy as np
import pytest
from pytest_mock import MockerFixture

from glcm_cupy import GLCM
from glcm_cupy.conf import *
from glcm_cupy.windowing import make_windows, im_shape_after_glcm
from tests.unit_tests import glcm_py
from tests.unit_tests.glcm_py_skimage import glcm_py_skimage


@pytest.mark.parametrize(
    "i",
    [
        np.asarray([0, ] * 9, dtype=np.uint8),
        np.asarray([1, ] * 9, dtype=np.uint8),
        np.asarray([255, ] * 9, dtype=np.uint8),
        np.asarray([0, 1, 2, 3, 4, 252, 253, 254, 255], dtype=np.uint8),
    ]
)
@pytest.mark.parametrize(
    "j",
    [
        np.asarray([0, ] * 9, dtype=np.uint8),
        np.asarray([1, ] * 9, dtype=np.uint8),
        np.asarray([255, ] * 9, dtype=np.uint8),
        np.asarray([0, 1, 2, 3, 4, 252, 253, 254, 255], dtype=np.uint8),
    ]
)
def test_glcm_from_windows(i, j):
    # We only test with 2 windows to reduce time taken.
    windows = 2
    g = GLCM(radius=1).run_ij(
        cp.asarray(np.tile(i, (windows, 1))),
        cp.asarray(np.tile(j, (windows, 1)))
    )

    # The sum of the values, since tiled, will be scaled by no of windows.
    actual = dict(
        homogeneity=float(g[..., HOMOGENEITY].sum() / windows),
        contrast=float(g[..., CONTRAST].sum() / windows),
        asm=float(g[..., ASM].sum() / windows),
        mean_i=float(g[..., MEAN_I].sum() / windows),
        mean_j=float(g[..., MEAN_J].sum() / windows),
        var_i=float(g[..., VAR_I].sum() / windows),
        var_j=float(g[..., VAR_J].sum() / windows),
        correlation=float(g[..., CORRELATION].sum() / windows)
    )

    expected = glcm_py(i, j, 256)
    assert actual == pytest.approx(expected)

    # The sum of the values, since tiled, will be scaled by no of windows.
    actual_skimage = dict(
        homogeneity=float(g[..., HOMOGENEITY].sum() / windows),
        contrast=float(g[..., CONTRAST].sum() / windows),
        asm=float(g[..., ASM].sum() / windows),
        correlation=float(g[..., CORRELATION].sum() / windows)
    )

    if (i == j).all():
        actual_skimage['correlation'] = 1

    expected_skimage = glcm_py_skimage(i, j)
    assert actual_skimage == pytest.approx(expected_skimage, abs=0.01)
