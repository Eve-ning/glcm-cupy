import cupy as cp
import pytest

from multikernel import glcm_module


@pytest.mark.parametrize(
    'max_value',
    [2, 10, 256]
)
@pytest.mark.parametrize(
    'no_of_values',
    [0, 1, 10, 100]
)
@pytest.mark.parametrize(
    'no_of_windows',
    [1, 2, 10, 100]
)
def test_mult_glcm(max_value, no_of_values, no_of_windows):
    """ Tests using a 3D Image """

    f = glcm_module.get_function('glcm_0')
    cp.random.seed(0)
    i = cp.random.randint(0, max_value, (no_of_windows, no_of_values), dtype=cp.uint8)
    j = cp.random.randint(0, max_value, (no_of_windows, no_of_values), dtype=cp.uint8)
    # i = cp.zeros((no_of_windows, no_of_values), dtype=cp.uint8)
    # j = cp.ones((no_of_windows, no_of_values), dtype=cp.uint8) * (max_value - 1)
    g = cp.zeros((no_of_windows, max_value, max_value), dtype=cp.uint8)
    ft = cp.zeros((no_of_windows, 8), dtype=cp.float32)

    f(
        grid=(1024, 1250),
        block=(512,),
        args=(
            i, j, max_value, no_of_values, no_of_windows, g, ft
        )
    )
    for w in range(no_of_windows):
        assert g.get()[w].sum() == no_of_values, str(g.get()[..., w])
