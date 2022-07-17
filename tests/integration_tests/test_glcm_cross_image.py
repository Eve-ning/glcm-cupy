import cupy as cp
import numpy as np
import pytest

from glcm_cupy import GLCMCross


def test_glcm_cross_image(ar_img_3d):
    ar = ar_img_3d[::20, ::20]
    g = GLCMCross(bin_to=16).run(ar)
    g_exp = np.load("expected/glcm_cross_image.npy")
    assert g == pytest.approx(g_exp, abs=1e-04)


def test_glcm_cross_image_cupy(ar_img_3d):
    ar = cp.asarray(ar_img_3d)[::20, ::20]
    g = GLCMCross(bin_to=16).run(ar)
    g_exp = np.load("expected/glcm_cross_image.npy")
    assert g.get() == pytest.approx(g_exp, abs=1e-04)
