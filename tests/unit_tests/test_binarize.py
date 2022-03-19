import cupy as cp
import numpy as np
import pytest

from glcm_cuda import GLCM
from tests.unit_tests import glcm_expected


@pytest.mark.parametrize(
    "bins",
    [2, 3, 256]
)
def test_binarize(i, j):
    g = GLCM.binarize(np.asarray([0, 1, 2], dtype=np.uint8), 2, 2)

    expected = glcm_expected(i, j)
    assert actual == pytest.approx(expected, abs=1e-2)

