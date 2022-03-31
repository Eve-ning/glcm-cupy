import numpy as np
from PIL import Image

from glcm_cupy.conf import ROOT_DIR
from glcm_cupy import GLCM


def test_image():
    img = Image.open(f"{ROOT_DIR}/data/image.jpg")
    ar = np.asarray(img)[::5, ::5]
    _ = GLCM(bin_to=16)._from_3dimage(ar)
