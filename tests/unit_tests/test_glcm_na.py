import numpy as np
import pytest

from glcm_cupy import GLCM


@pytest.mark.parametrize(
    'inner_size',
    (3, 5, 7)
)
@pytest.mark.parametrize(
    'na_pad',
    (1, 2, 3)
)
@pytest.mark.parametrize(
    'bins',
    (16,)
)
def test_glcm_na_padding(inner_size, na_pad, bins):
    """ Tests effects of NA Padding

    Examples:
        A 3x3 == 5x5 (1 NA border padding)

        # # # # #
        # 0 1 2 #      0 1 2
        # 3 4 5 #  ==  3 4 5
        # 6 7 8 #      6 7 8
        # # # # #

    """
    outer_size = inner_size + na_pad * 2
    ar_inner = np.random.randint(0, bins, (inner_size, inner_size, 1))
    ar_outer = np.empty([outer_size, outer_size, 1])
    ar_outer[:] = np.nan
    ar_outer[na_pad:-na_pad, na_pad:-na_pad] = ar_inner
    glcm = GLCM(radius=0, bin_from=bins, bin_to=bins)
    ar_outer_g = glcm.run(ar_outer)
    ar_inner_g = glcm.run(ar_inner)
    assert (ar_outer_g[na_pad:-na_pad, na_pad:-na_pad] == ar_inner_g).all()


@pytest.mark.parametrize(
    'size',
    (5, 7, 9)
)
@pytest.mark.parametrize(
    'holes',
    (1, 3, 5)
)
def test_glcm_na_holes(size, holes):
    """ Tests effects of NA holes in images

    Examples:
        For an image with only 1 number, for MOST combination of holes, the
        resulting GLCM is the same as the one without holes.

        0 0 0      # 0 0      0 # 0
        0 0 0  ==  0 # 0  ==  0 # #
        0 0 0      # 0 0      # 0 #

        This is because only probability is considered, NAs are not counted in
        the denominator.

        However, if the windows spans an all NA case, then it'll be different.

        To avoid this, we limit the number of random holes

    """
    bins = 4
    hole_ixs = np.random.choice(size * size, holes, replace=False)
    ar = np.ones((size * size, 1)) * (bins - 1)
    ar_holes = ar.copy()
    ar_holes[hole_ixs] = np.nan
    ar = ar.reshape(size, size, 1)
    ar_holes = ar_holes.reshape(size, size, 1)
    glcm = GLCM(radius=1, bin_from=bins, bin_to=bins)
    ar_g = glcm.run(ar)
    ar_holes_g = glcm.run(ar_holes)
    assert (ar_g == ar_holes_g).all()
