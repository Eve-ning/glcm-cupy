import cupy as cp
import numpy as np
import pytest

from glcm_cuda import GLCM


def test_from_3dimage(np_array_3d):
    """ Tests using a 3D Image """
    GLCM().from_3dimage(np_array_3d)

def test_from_2dimage(np_array_2d):
    """ Tests with a 2D Image (1 Channel) """
    GLCM().from_2dimage(np_array_2d)

def test_from_windows(np_array_1d):
    """ Tests the most atomic function _from_windows()

    Notes:
        This is not to be used directly, thus a private function.

    """
    GLCM()._from_windows(np_array_1d, np_array_1d)

def test_bad_dtype(np_array_1d):
    """ Using dtype != np.uint8 should raise ValueError

    Notes:
        _from_windows is the lowest level function that enters the GLCM.
        So this will catch all dtype issues.

    """
    with pytest.raises(ValueError):
        GLCM()._from_windows(np_array_1d.astype(int), np_array_1d)
    with pytest.raises(ValueError):
        GLCM()._from_windows(np_array_1d, np_array_1d.astype(int))
    with pytest.raises(ValueError):
        GLCM()._from_windows(np_array_1d.astype(int), np_array_1d.astype(int))
