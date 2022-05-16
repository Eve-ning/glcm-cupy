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
@pytest.mark.parametrize(
    "ix_combos",
    [
        [[0, 1]],
        [[1, 2]],
        [[0, 1], [1, 2]],
        None
    ]
)
def test_cross_glcm(size, bins, radius, ix_combos):
    ar = np.random.randint(0, bins, [size, size, 3])
    g = GLCMCross(radius=radius, bin_from=bins, bin_to=bins,
                  ix_combos=ix_combos).run(ar)
    g_fn = glcm_cross(ar, radius=radius, bin_from=bins, bin_to=bins,
                      ix_combos=ix_combos)
    expected = glcm_cross_py_im(ar, radius=radius,
                                bin_from=bins, bin_to=bins,
                                ix_combos=ix_combos)
    assert g == pytest.approx(expected, abs=0.001)
    assert g_fn == pytest.approx(expected, abs=0.001)
