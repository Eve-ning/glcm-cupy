import numpy as np
import pytest

from glcm_cupy.cross.glcm_cross import GLCMCross, glcm_cross
from glcm_cupy.cross.glcm_cross_py import glcm_cross_py_im


@pytest.mark.parametrize(
    "size",
    [9, ]
)
@pytest.mark.parametrize(
    "bins",
    [4, 16]
)
@pytest.mark.parametrize(
    "radius",
    [1, 2, 4]
)
def test_cross_glcm(size, bins, radius):
    # We only test with 2 windows to reduce time taken.
    ar = np.random.randint(0, bins, [size, size, 2])
    g = GLCMCross(radius=radius, bin_from=bins, bin_to=bins).run(ar)
    g_fn = glcm_cross(ar, radius=radius, bin_from=bins, bin_to=bins)
    expected = glcm_cross_py_im(ar, radius=radius, bin_from=bins, bin_to=bins)
    assert g == pytest.approx(expected, abs=0.001)
    assert g_fn == pytest.approx(expected, abs=0.001)
