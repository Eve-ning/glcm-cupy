import os

import pytest
import os

import numpy as np
import cupy as cp
import pytest
from cupyx.profiler import benchmark

from glcm_cuda import GLCM

TEST_SIZE = 25


@pytest.fixture()
def np_array_3d():
    return np.random.randint(0, 256, (TEST_SIZE, TEST_SIZE, 3), dtype=np.uint8)


@pytest.fixture()
def np_array_2d():
    return np.random.randint(0, 256, (TEST_SIZE, TEST_SIZE), dtype=np.uint8)

@pytest.fixture()
def np_array_1d():
    return np.random.randint(0, 256, (TEST_SIZE, ), dtype=np.uint8)
