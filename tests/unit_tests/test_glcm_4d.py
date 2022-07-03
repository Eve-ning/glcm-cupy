import cupy as cp
import numpy as np
import pytest

from glcm_cupy import GLCM


@pytest.mark.parametrize(
    "size",
    [15, ]
)
@pytest.mark.parametrize(
    "bins",
    [4, 16]
)
@pytest.mark.parametrize(
    "radius",
    [1, 2, 4]
)
@pytest.mark.parametrize(
    "batches",
    [1, 2, 10]
)
def test_glcm_4d(size, bins, radius, batches):
    """ Tests the case where a batch is sent. """
    np.random.seed(0)
    ar_batch = np.random.randint(0, bins, [batches, size, size, 1])
    glcm = GLCM(radius=radius, bin_from=bins, bin_to=bins)
    g_batch = glcm.run(ar_batch)
    g = np.stack([glcm.run(ar) for ar in ar_batch])
    assert g == pytest.approx(g_batch, abs=1e-06)


@pytest.mark.parametrize(
    "size",
    [15, ]
)
@pytest.mark.parametrize(
    "bins",
    [4, 16]
)
@pytest.mark.parametrize(
    "radius",
    [1, 2, 4]
)
@pytest.mark.parametrize(
    "batches",
    [1, 2, 10]
)
def test_glcm_4d(size, bins, radius, batches):
    """ Tests the case where a batch is sent. """
    cp.random.seed(0)
    ar_batch = cp.random.randint(0, bins, [batches, size, size, 1])
    glcm = GLCM(radius=radius, bin_from=bins, bin_to=bins)
    g_batch = glcm.run(ar_batch)
    g = cp.stack([glcm.run(ar) for ar in ar_batch])
    assert g.get() == pytest.approx(g_batch.get(), abs=1e-06)
