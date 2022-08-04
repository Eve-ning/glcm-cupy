import numpy as np

from glcm_cupy import GLCM


def test_glcm_na():
    """ A 3x3 == 5x5 (1 NA border padding)

    # # # # #
    # 0 1 2 #      0 1 2
    # 3 4 5 #  ==  3 4 5
    # 6 7 8 #      6 7 8
    # # # # #

    """
    inner_size = 3
    na_pad = 1
    outer_size = inner_size + na_pad * 2
    bins = 16
    ar_inner = np.random.randint(0, bins, (inner_size, inner_size, 1))
    ar_outer = np.empty([outer_size, outer_size, 1])
    ar_outer[:] = np.nan
    ar_outer[na_pad:-na_pad, na_pad:-na_pad] = ar_inner
    glcm = GLCM(radius=0, bin_from=bins, bin_to=bins)
    ar_outer_g = glcm.run(ar_outer)
    ar_inner_g = glcm.run(ar_inner)
    assert (ar_outer_g[na_pad:-na_pad, na_pad:-na_pad] == ar_inner_g).all()
