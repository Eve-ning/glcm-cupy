from typing import Tuple

import numpy as np
import pytest

from glcm_cupy import GLCM
from glcm_cupy.windowing import make_windows, im_shape_after_glcm


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

    im_shape_after = im_shape_after_glcm(im_shape, step_size, radius)

    if im_shape_after[0] <= 0 or im_shape_after[1] <= 0:
        # If the make windows is invalid, we assert that it throws an error
        with pytest.raises(ValueError):
            make_windows(im, radius, step_size)
    else:
        # Else, we assert the correct shape returns
        windows = make_windows(im, radius, step_size)
        assert (np.prod(im_shape_after),
                (radius * 2 + 1) ** 2) == windows[0][0].shape


@pytest.mark.parametrize(
    "bins",
    [1, 2, 256]
)
def test_glcm_binner(bins):
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