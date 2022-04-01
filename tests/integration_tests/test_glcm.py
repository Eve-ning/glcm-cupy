import numpy as np

from glcm_cupy import GLCM


def test_from_3dimage(ar_3d):
    """ Tests using a 3D Image """
    GLCM().run(ar_3d)


def test_from_2dimage(ar_2d):
    """ Tests with a 2D Image (1 Channel) """
    GLCM().run(ar_2d)


def test__from_windows():
    """ Tests the protected method _from_windows

    Notes:
        This uses a protected class, thus has important constraints to follow.
        This requires an additional axis at the front.

    """
    ar_0 = np.random.randint(0, 100, 10, dtype=np.uint8)
    ar_1 = np.random.randint(0, 100, 10, dtype=np.uint8)
    GLCM().run_ij(ar_0[..., np.newaxis], ar_1[..., np.newaxis])
