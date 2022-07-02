import cupy as cp
import numpy as np
import pytest
from PIL import Image

from glcm_cupy import GLCM
from glcm_cupy.conf import ROOT_DIR


def test_glcm_image():
    img = Image.open(f"{ROOT_DIR}/data/image.jpg")
    ar = np.asarray(img)[::20, ::20]
    g = GLCM(bin_to=16).run(ar)
    g_exp = np.load("expected/glcm_image.npy")
    assert g == pytest.approx(g_exp, abs=1e-06)


def test_glcm_image_cupy():
    img = Image.open(f"{ROOT_DIR}/data/image.jpg")
    ar = cp.asarray(img)[::20, ::20]
    g = GLCM(bin_to=16).run(ar)
    g_exp = np.load("expected/glcm_image.npy")
    assert g.get() == pytest.approx(g_exp, abs=1e-06)
