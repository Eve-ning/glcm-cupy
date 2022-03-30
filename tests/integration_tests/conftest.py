import numpy as np
import pytest
from PIL import Image

from glcm_pycuda.definitions import ROOT_DIR

TEST_SIZE = 25


@pytest.fixture()
def ar_3d():
    return np.random.randint(0, 256, (TEST_SIZE, TEST_SIZE, 3), dtype=np.uint8)


@pytest.fixture()
def ar_img_3d():
    return Image.open(f"{ROOT_DIR}/data/image.jpg")


@pytest.fixture()
def ar_2d():
    return np.random.randint(0, 256, (TEST_SIZE, TEST_SIZE), dtype=np.uint8)


@pytest.fixture()
def ar_1d():
    return np.random.randint(0, 256, (TEST_SIZE,), dtype=np.uint8)
