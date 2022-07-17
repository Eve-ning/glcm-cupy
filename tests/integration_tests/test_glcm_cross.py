import inspect
from pathlib import Path

import numpy as np
import pytest

from glcm_cupy import GLCMCross, glcm_cross

THIS_DIR = Path(__file__).parent


def test_from_3dimage(ar_3d):
    """ Tests using a 3D Image """
    g = GLCMCross().run(ar_3d)
    g_exp = np.load(THIS_DIR / "expected/ar_3d_glcm_cross.npy")
    assert g == pytest.approx(g_exp, abs=1e-04)


def test_from_3dimage_ix_combo(ar_3d):
    """ Tests using a 3D Image with selected ix_combos """
    g = GLCMCross(ix_combos=[[0, 1], [1, 2]]).run(ar_3d)
    g_exp = np.load(THIS_DIR / "expected/ar_3d_combo_glcm_cross.npy")
    assert g == pytest.approx(g_exp, abs=1e-04)
    assert g.shape[2] == 2


def test_from_2dimage(ar_2d):
    """ Tests with a 2D Image (1 Channel) """

    # This is not possible as we need > 1 channel to cross
    with pytest.raises(ValueError):
        GLCMCross().run(ar_2d[..., np.newaxis])


def test_output_match(ar_3d):
    """ Tests if class & function outputs match """
    assert GLCMCross().run(ar_3d) == pytest.approx(glcm_cross(ar_3d),
                                                   abs=1e-04)


def test_from_3dimage_cp(ar_3d_cp):
    """ Tests using a 3D Image """
    g = GLCMCross().run(ar_3d_cp)
    g_exp = np.load(THIS_DIR / "expected/ar_3d_glcm_cross.npy")
    assert g.get() == pytest.approx(g_exp, abs=1e-04)


def test_from_3dimage_ix_combo_cp(ar_3d_cp):
    """ Tests using a 3D Image with selected ix_combos """
    g = GLCMCross(ix_combos=[[0, 1], [1, 2]]).run(ar_3d_cp)
    g_exp = np.load(THIS_DIR / "expected/ar_3d_combo_glcm_cross.npy")
    assert g.get() == pytest.approx(g_exp, abs=1e-04)
    assert g.shape[2] == 2


def test_from_2dimage_cp(ar_2d_cp):
    """ Tests with a 2D Image (1 Channel) """

    # This is not possible as we need > 1 channel to cross
    with pytest.raises(ValueError):
        GLCMCross().run(ar_2d_cp[..., np.newaxis])


def test_output_match_cp(ar_3d_cp):
    """ Tests if class & function outputs match """
    # XXX: pytest.approx does not support CuPy.
    # It needs to get the NumPy array instead.
    assert GLCMCross().run(ar_3d_cp).get() == \
           pytest.approx(glcm_cross(ar_3d_cp).get(), abs=1e-04)


def test_signature_match():
    """ Tests if class & function signatures match """
    cls = dict(inspect.signature(GLCMCross).parameters)
    fn = dict(inspect.signature(glcm_cross).parameters)
    del fn['im']
    assert cls == fn
