import cupy as cp
import numpy as np
import pytest

from glcm_cupy import GLCM, glcm


def test_glcm_image(ar_img_3d):
    ar = ar_img_3d[::20, ::20]
    g = GLCM(bin_to=16).run(np.stack([ar, ar]))
    g0, g1 = g[0], g[1]
    g = GLCM(bin_to=16).run(ar)
    g_exp = np.load("expected/glcm_image.npy")
    assert g == pytest.approx(g_exp, abs=1e-06)
    assert g0 == pytest.approx(g_exp, abs=1e-06)
    assert g1 == pytest.approx(g_exp, abs=1e-06)


def test_glcm_image_cupy(ar_img_3d):
    ar = cp.asarray(ar_img_3d)[::20, ::20]
    g = GLCM(bin_to=16).run(cp.stack([ar, ar]))
    g0, g1 = g[0], g[1]
    g_exp = np.load("expected/glcm_image.npy")
    assert g0.get() == pytest.approx(g_exp, abs=1e-06)
    assert g1.get() == pytest.approx(g_exp, abs=1e-06)
