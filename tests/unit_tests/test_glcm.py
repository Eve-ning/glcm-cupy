import numpy as np
import cupy as cp
import pytest

from glcm_cuda import GLCM

@pytest.mark.parametrize(
    "i,j",
    [
        [
            cp.asarray([0,0,0,0]),cp.asarray([1,1,1,1]),

        ]
    ]
)
def test_glcm_asm(i, j):
    g = GLCM()._from_windows(i, j)
    assert g[..., GLCM.ASM].sum() == 1