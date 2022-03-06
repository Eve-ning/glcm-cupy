import os

import numpy as np
import pytest
from PIL import Image

from definitions import ROOT_DIR
from glcm_cuda import GLCM

@pytest.fixture(autouse=True)
def set_env():
    os.environ['CUPY_EXPERIMENTAL_SLICE_COPY'] = '1'

def test_image():
    img = Image.open(f"{ROOT_DIR}/data/image.jpg")
    ar = np.asarray(img)
    g = GLCM().from_3dimage(ar)
