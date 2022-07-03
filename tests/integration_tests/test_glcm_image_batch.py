import cupy as cp
import numpy as np
import pytest
from matplotlib.image import imread

from glcm_cupy import GLCM
from glcm_cupy.conf import ROOT_DIR


def test_glcm_image():
    img = imread(f"{ROOT_DIR}/data/image.jpg")
    ar = img[::20, ::20]
    g = GLCM(bin_to=16).run(np.stack([ar, ar]))
    g0, g1 = g[0], g[1]
    g = GLCM(bin_to=16).run(ar)
    g_exp = np.load("expected/glcm_image.npy")
    assert g == pytest.approx(g_exp, abs=1e-06)
    assert g0 == pytest.approx(g_exp, abs=1e-06)
    assert g1 == pytest.approx(g_exp, abs=1e-06)


def test_glcm_image_cupy():
    img = imread(f"{ROOT_DIR}/data/image.jpg")
    ar = cp.asarray(img)[::20, ::20]
    g = GLCM(bin_to=16).run(cp.stack([ar, ar]))
    g0, g1 = g[0], g[1]
    g_exp = np.load("expected/glcm_image.npy")
    assert g0.get() == pytest.approx(g_exp, abs=1e-06)
    assert g1.get() == pytest.approx(g_exp, abs=1e-06)
