import numpy as np
import pytest
from tifffile import tifffile

from definitions import ROOT_DIR
from tests.unit_tests import glcm_expected


@pytest.mark.parametrize(
    "tree_name,x,y,w,h",
    [
        ["Campnosperma", 3734, 704, 241, 211],
        ["Spathodea", 2603, 1561, 289, 341]
    ]
)
@pytest.mark.parametrize(
    "bins",
    [8, 16]
)
@pytest.mark.parametrize(
    "step_size",
    [1, 2]
)
def test_image(tree_name, x, y, w, h, bins, step_size):
    ar = tifffile.imread(f"{ROOT_DIR}/data/result_NIR.tif") \
        [y:y + h, x:x + w]
    ar = ((ar / 2 ** 14) * bins).astype(np.uint8)
    g = glcm_expected(ar[:-step_size], ar[step_size:])
    print(g)
