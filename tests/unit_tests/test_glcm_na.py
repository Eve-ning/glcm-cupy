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
    ar_inner = np.asarray([[0, 1, 0], [1, 0, 1], [1, 1, 1]])[..., np.newaxis]
    ar_outer = np.empty([5, 5, 1])
    ar_outer[:] = np.nan
    ar_outer[1:-1, 1:-1] = ar_inner
    glcm = GLCM(radius=0, bin_from=2, bin_to=2)
    ar_outer_g = glcm.run(ar_outer)
    ar_inner_g = glcm.run(ar_inner)
    assert (ar_outer_g[1, 1, 0] == ar_inner_g[0, 0, 0]).all()
