from typing import Tuple

import cupy as cp
import numpy as np
import pytest
from pytest_mock import MockerFixture

from glcm_cupy import GLCM
from glcm_cupy.conf import *
from tests.unit_tests import glcm_py


@pytest.mark.parametrize(
    "i",
    [
        np.asarray([0, ] * 9, dtype=np.uint8),
        np.asarray([1, ] * 9, dtype=np.uint8),
        np.asarray([255, ] * 9, dtype=np.uint8),
        np.asarray([0, 1, 2, 3, 4, 252, 253, 254, 255], dtype=np.uint8),
    ]
)
@pytest.mark.parametrize(
    "j",
    [
        np.asarray([0, ] * 9, dtype=np.uint8),
        np.asarray([1, ] * 9, dtype=np.uint8),
        np.asarray([255, ] * 9, dtype=np.uint8),
        np.asarray([0, 1, 2, 3, 4, 252, 253, 254, 255], dtype=np.uint8),
    ]
)
def test_glcm_from_windows(i, j):
    # We only test with 2 windows to reduce time taken.
    windows = 2
    g = GLCM(radius=1)._from_windows(
        cp.asarray(np.tile(i, (windows, 1))),
        cp.asarray(np.tile(j, (windows, 1)))
    )

    # The sum of the values, since tiled, will be scaled by no of windows.
    actual = dict(
        homogeneity=float(g[..., HOMOGENEITY].sum() / windows),
        contrast=float(g[..., CONTRAST].sum() / windows),
        asm=float(g[..., ASM].sum() / windows),
        mean_i=float(g[..., MEAN_I].sum() / windows),
        mean_j=float(g[..., MEAN_J].sum() / windows),
        var_i=float(g[..., VAR_I].sum() / windows),
        var_j=float(g[..., VAR_J].sum() / windows),
        correlation=float(g[..., CORRELATION].sum() / windows)
    )

    expected = glcm_py(i, j)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "im_shape",
    [
        (1, 1),
        (5, 5),
        (10, 10),
        (20, 20),
        (100, 100),
    ]
)
@pytest.mark.parametrize(
    "radius",
    [1, 2, 3, 4]
)
@pytest.mark.parametrize(
    "step_size",
    [1, 2, 3, 4]
)
def test_glcm_make_windows(
    im_shape: Tuple[int, int],
    radius: int,
    step_size: int
):
    """ Test Make Window returns for various image sizes

    Args:
        im_shape: Shape of the input image
        radius: Radius of each window
        step_size: Step Size distance between windows
    """
    im = np.zeros(im_shape, dtype=np.uint8)

    y_windows = (im_shape[0] - step_size - radius * 2)
    x_windows = (im_shape[1] - step_size - radius * 2)
    y_cells = radius * 2 + 1
    x_cells = radius * 2 + 1

    if y_windows <= 0 or x_windows <= 0:
        # If the make windows is invalid, we assert that it throws an error
        with pytest.raises(ValueError):
            GLCM._make_windows(im, radius * 2 + 1, step_size)
    else:
        # Else, we assert the correct shape returns
        windows = GLCM._make_windows(im, radius * 2 + 1, step_size)
        assert (x_windows * y_windows, x_cells * y_cells) == windows[0].shape


@pytest.mark.parametrize(
    "windows",
    [1,
     MAX_PARTITION_SIZE,
     MAX_PARTITION_SIZE + 1,
     MAX_PARTITION_SIZE * 2,
     MAX_PARTITION_SIZE * 2 + 1]
)
def test_glcm_partition(
    windows: int,
    mocker: MockerFixture
):
    """ Verifies the input sizes during partitioning

    Notes:
        If partition size is 10, and we have 25 windows
        We expect the input list to be 10, 10, 5

        This is done to reduce to GLCM size input into the GPU Kernel

    Args:
        windows: Number of windows
    """
    glcm_instance = GLCM()
    WINDOW_SIZE = 1

    class MockFeatures:
        """ Mocks the np.ndarray to do nothing """

        def reshape(self, *args, **kwargs): ...

        def __setitem__(self, key, value): ...

    mocker.patch('cupy.zeros', return_value=MockFeatures())
    mocker.patch.object(
        glcm_instance, '_make_windows',
        return_value=(
            np.zeros((windows, WINDOW_SIZE), dtype=np.uint8),
            np.zeros((windows, WINDOW_SIZE), dtype=np.uint8),
        )
    )
    mock_from_windows = \
        mocker.patch.object(glcm_instance, '_from_windows', return_value=[])
    glcm_instance._from_2dimage(np.asarray([[]]))

    # The LHS extracts the array size calls to _from_windows
    #  We expect the windows to be split to chunks of PARTITION_SIZE
    # The RHS simulates that action
    #  The first list comprehension copies the PARTITION_SIZE depends on the
    #  quotient
    #  The second is the remainder, however 0 will produce a [] as expected.
    assert [call[0][0].shape[0] for call in
            mock_from_windows.call_args_list] == \
           [MAX_PARTITION_SIZE for _ in range(windows // MAX_PARTITION_SIZE)] + \
           ([windows % MAX_PARTITION_SIZE] if windows % MAX_PARTITION_SIZE else [])


@pytest.mark.parametrize(
    "bins",
    [1, 2, 256]
)
def test_glcm_binarize(bins):
    """ Test binarizing of an image

    Notes:
        The result bin value is not achievable.
        Thus, the maximum value for 256 is 255.

        Note that 255 can only happen on a shrinkage
        E.g.
        0, 1, 2 [bins=3] -> 0, 85, 170 [bins=256]

    Args:
        bins: The result bins
    """
    g = GLCM._binner(np.asarray([0, 1, 2], dtype=np.uint8), 3, bins)

    assert g[0] == 0 // 3
    assert g[1] == bins // 3
    assert g[2] == bins * 2 // 3
