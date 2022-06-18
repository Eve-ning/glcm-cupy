import inspect

import numpy as np
import pytest

from glcm_cupy import GLCM, glcm


def test_from_3dimage(ar_3d):
    """ Tests using a 3D Image """
    GLCM().run(ar_3d)


def test_from_2dimage(ar_2d):
    """ Tests with a 2D Image (1 Channel) """
    GLCM().run(ar_2d[..., np.newaxis])


def test_output_match(ar_3d):
    """ Tests if class & function outputs match """
    assert GLCM().run(ar_3d) == pytest.approx(glcm(ar_3d))


def test_from_3dimage_cp(ar_3d_cp):
    """ Tests using a 3D Image """
    GLCM().run(ar_3d_cp)


def test_from_2dimage_cp(ar_2d_cp):
    """ Tests with a 2D Image (1 Channel) """
    GLCM().run(ar_2d_cp[..., np.newaxis])


def test_output_match_cp(ar_3d_cp):
    """ Tests if class & function outputs match """
    # XXX: pytest.approx does not support CuPy.
    # It needs to get the NumPy array instead.
    assert GLCM().run(ar_3d_cp).get() == pytest.approx(glcm(ar_3d_cp).get())

def test_signature_match():
    """ Tests if class & function signatures match """
    cls = dict(inspect.signature(GLCM).parameters)
    fn = dict(inspect.signature(glcm).parameters)
    del fn['im']
    assert cls == fn
