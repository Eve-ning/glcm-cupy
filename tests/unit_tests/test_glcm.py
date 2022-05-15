import numpy as np
import pytest

from glcm_cupy import GLCM, glcm
from glcm_cupy.glcm.glcm_py import glcm_py_3d


@pytest.mark.parametrize(
    "size",
    [15, ]
)
@pytest.mark.parametrize(
    "bins",
    [4, 16]
)
@pytest.mark.parametrize(
    "radius",
    [1, 2, 4]
)
def test_glcm(size, bins, radius):
    ar = np.random.randint(0, bins, [size, size, 1])
    g = GLCM(radius=radius, bin_from=bins, bin_to=bins).run(ar)
    g_fn = glcm(ar, radius=radius, bin_from=bins, bin_to=bins)
    expected = glcm_py_3d(ar, radius=radius, bin_from=bins, bin_to=bins)
    assert g == pytest.approx(expected, abs=0.001)
    assert g_fn == pytest.approx(expected, abs=0.001)
