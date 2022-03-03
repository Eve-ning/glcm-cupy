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
def test_glcm(
        i, j,
        homogeneity,
        contrast,
        asm,
        mean_i,
        mean_j,
        var_i,
        var_j,
        correlation
):
    g = GLCM()._from_windows(i, j)

    assert g[..., GLCM.HOMOGENEITY].sum() == homogeneity
    assert g[..., GLCM.CONTRAST].sum() == contrast
    assert g[..., GLCM.ASM].sum() == asm
    assert g[..., GLCM.MEAN_I].sum() == mean_i
    assert g[..., GLCM.MEAN_J].sum() == mean_j
    assert g[..., GLCM.VAR_I].sum() == var_i
    assert g[..., GLCM.VAR_J].sum() == var_j
    assert g[..., GLCM.CORRELATION].sum() == correlation
