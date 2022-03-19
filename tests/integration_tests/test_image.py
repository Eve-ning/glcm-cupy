import os

import numpy as np
import pytest
from PIL import Image

from definitions import ROOT_DIR
from glcm_cuda import GLCM

@pytest.mark.skip("Image takes too long to test")
def test_image():
    img = Image.open(f"{ROOT_DIR}/data/image.jpg")
    ar = np.asarray(img)[::5]
    g = GLCM().from_3dimage(ar)
