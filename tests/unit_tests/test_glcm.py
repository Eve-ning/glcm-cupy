import cupy as cp
import numpy as np
import pytest

from glcm_cuda import GLCM
from tests.unit_tests import glcm_expected


@pytest.mark.parametrize(
    "i",
    [
        np.asarray([0, 0, 0, 0], dtype=np.uint8),
        np.asarray([1, 1, 1, 1], dtype=np.uint8),
        np.asarray([255, 255, 255, 255], dtype=np.uint8),
        np.asarray([0, 1, 254, 255], dtype=np.uint8),
    ]
)
@pytest.mark.parametrize(
    "j",
    [
        np.asarray([0, 0, 0, 0], dtype=np.uint8),
        np.asarray([1, 1, 1, 1], dtype=np.uint8),
        np.asarray([255, 255, 255, 255], dtype=np.uint8),
        np.asarray([0, 1, 254, 255], dtype=np.uint8),
    ]
)
def test_glcm(i, j):
    g = GLCM()._from_windows(cp.asarray(i),
                             cp.asarray(j))
    actual = dict(
        homogeneity=float(g[..., GLCM.HOMOGENEITY].sum()),
        contrast=float(g[..., GLCM.CONTRAST].sum()),
        asm=float(g[..., GLCM.ASM].sum()),
        mean_i=float(g[..., GLCM.MEAN_I].sum()),
        mean_j=float(g[..., GLCM.MEAN_J].sum()),
        var_i=float(g[..., GLCM.VAR_I].sum()),
        var_j=float(g[..., GLCM.VAR_J].sum()),
        correlation=float(g[..., GLCM.CORRELATION].sum())
    )

    expected = glcm_expected(i, j)
    assert pytest.approx(expected) == actual

