import inspect

import numpy as np
import pytest

from glcm_cupy import GLCMCross, glcm_cross


def test_from_3dimage(ar_3d):
    """ Tests using a 3D Image """
    GLCMCross().run(ar_3d)


def test_from_3dimage_ix_combo(ar_3d):
    """ Tests using a 3D Image with selected ix_combos """
    g = GLCMCross(ix_combos=[[0, 1], [1, 2]]).run(ar_3d)
    assert g.shape[2] == 2


def test_from_2dimage(ar_2d):
    """ Tests with a 2D Image (1 Channel) """

    # This is not possible as we need > 1 channel to cross
    with pytest.raises(ValueError):
        GLCMCross().run(ar_2d[..., np.newaxis])


def test_output_match(ar_3d):
    """ Tests if class & function outputs match """
    assert GLCMCross().run(ar_3d) == pytest.approx(glcm_cross(ar_3d))


def test_signature_match():
    """ Tests if class & function signatures match """
    cls = dict(inspect.signature(GLCMCross).parameters)
    fn = dict(inspect.signature(glcm_cross).parameters)
    del fn['im']
    assert cls == fn
