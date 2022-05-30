import numpy as np
from PIL import Image

from glcm_cupy import GLCM
from glcm_cupy.conf import ROOT_DIR


def test_glcm_image():
    img = Image.open(f"{ROOT_DIR}/data/image.jpg")
    ar = np.asarray(img)[::5, ::5]
    g = GLCM(bin_to=16).run(ar)
