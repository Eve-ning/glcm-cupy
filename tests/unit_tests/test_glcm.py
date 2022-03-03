import cupy as cp
import pytest

from .test_cases import *

from glcm_cuda import GLCM




@pytest.mark.parametrize(
    "i,j,homogeneity,contrast,asm,mean_i,mean_j,var_i,var_j,correlation",
    [
        simple_0, simple_1
    ]
)
def test_glcm_asm(i, j):
    g = GLCM()._from_windows(i, j)
    assert g[..., GLCM.ASM].sum() == 1