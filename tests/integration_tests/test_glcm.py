import numpy as np

from glcm_cupy import GLCM


def test_from_3dimage(ar_3d):
    """ Tests using a 3D Image """
    GLCM().run(ar_3d)


def test_from_2dimage(ar_2d):
    """ Tests with a 2D Image (1 Channel) """
    GLCM().run(ar_2d[..., np.newaxis])
