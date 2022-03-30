import numpy as np
from PIL import Image

from glcm_pycuda.definitions import ROOT_DIR
from glcm_pycuda import GLCM


def test_image():
    img = Image.open(f"{ROOT_DIR}/data/image.jpg")
    ar = np.asarray(img)[::5]
    _ = GLCM().from_3dimage(ar)
