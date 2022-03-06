import os

import numpy as np
import cupy as cp
import pytest
from cupyx.profiler import benchmark

from glcm_cuda import GLCM

TEST_SIZE = 100

@pytest.fixture(autouse=True)
def set_env():
    os.environ['CUPY_EXPERIMENTAL_SLICE_COPY'] = '1'

def test_benchmark_3dimage(set_env):
    """ Simply benchmarks the GLCM

    Returns:

    """
    GLCM(bins=16).from_3dimage(
        np.random.randint(0, 256, (TEST_SIZE, TEST_SIZE, 3), dtype=np.uint8)
    )


def test_benchmark_2dimage():
    """ Simply benchmarks the GLCM

    Returns:

    """
    GLCM().from_2dimage(
        np.random.randint(0, 256, (TEST_SIZE, TEST_SIZE), dtype=np.uint8))


def test_benchmark_ij():
    """ Simply benchmarks the GLCM

    Returns:

    """
    # Simulates a 16x16 window
    b = benchmark(GLCM()._from_windows, (
        cp.random.randint(0, 256, TEST_SIZE, dtype=np.uint8),
        cp.random.randint(0, 256, TEST_SIZE, dtype=np.uint8)
    ), n_repeat=100)
    print(np.mean(b.cpu_times) * 2000 * 2000 / 16 / 16 * 3)
