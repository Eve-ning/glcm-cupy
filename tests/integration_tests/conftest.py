import cupy as cp
import numpy as np
import pytest
from matplotlib.image import imread

from glcm_cupy.conf import ROOT_DIR

TEST_SIZE = 25


@pytest.fixture()
def ar_3d():
    np.random.seed(8008)
    return np.random.randint(0, 256, (TEST_SIZE, TEST_SIZE, 3), dtype=np.uint8)


@pytest.fixture()
def ar_3d_cp(ar_3d):
    return cp.asarray(ar_3d)


@pytest.fixture()
def ar_img_3d():
    return imread(f"{ROOT_DIR}/data/image.jpg")


@pytest.fixture()
def ar_2d():
    np.random.seed(8008)
    return np.random.randint(0, 256, (TEST_SIZE, TEST_SIZE), dtype=np.uint8)


@pytest.fixture()
def ar_1d():
    np.random.seed(8008)
    return np.random.randint(0, 256, (TEST_SIZE,), dtype=np.uint8)


@pytest.fixture()
def ar_2d_cp(ar_2d):
    return cp.asarray(ar_2d)


@pytest.fixture()
def ar_1d_cp():
    np.random.seed(8008)
    return cp.random.randint(0, 256, (TEST_SIZE,), dtype=cp.uint8)
