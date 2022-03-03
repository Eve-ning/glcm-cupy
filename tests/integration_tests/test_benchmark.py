import numpy as np
import cupy as cp
from glcm_cuda import GLCM

TEST_SIZE = 20

def test_benchmark_3dimage():
    """ Simply benchmarks the GLCM

    Returns:

    """
    g = GLCM().from_3dimage(
        np.random.randint(0, 256, (TEST_SIZE, TEST_SIZE, 3), dtype=int)
    )


def test_benchmark_2dimage():
    """ Simply benchmarks the GLCM

    Returns:

    """
    g = GLCM().from_2dimage(np.random.randint(0, 256, (TEST_SIZE, TEST_SIZE), dtype=int))


def test_benchmark_ij():
    """ Simply benchmarks the GLCM

    Returns:

    """
    g = GLCM()._from_windows(
        i=cp.random.randint(0, 256, TEST_SIZE, dtype=int),
        j=cp.random.randint(0, 256, TEST_SIZE, dtype=int)
    )
