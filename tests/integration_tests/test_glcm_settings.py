import pytest

from glcm_cuda import GLCM


def _test_from_windows(glcm: GLCM, np_array_1d):
    """ Tests the most atomic function _from_windows()

    Notes:
        This is not to be used directly, thus a private function.

    """
    glcm._from_windows(np_array_1d, np_array_1d)


@pytest.mark.parametrize('max_value', [-1, 0, 256, 257])
def test_max_value(np_array_1d, max_value):
    """ Tests Max Value Setting """
    with pytest.raises(ValueError):
        _test_from_windows(GLCM(max_value=max_value), np_array_1d)


@pytest.mark.parametrize('bins', [-1, 0, 1, 256, 257])
def test_bins(np_array_1d, bins):
    """ Tests Bins Setting """
    with pytest.raises(ValueError):
        _test_from_windows(GLCM(bins=bins), np_array_1d)


@pytest.mark.parametrize('bins', [-1, 0])
def test_step_size(np_array_1d, bins):
    """ Tests Step Size Setting """
    with pytest.raises(ValueError):
        _test_from_windows(GLCM(bins=bins), np_array_1d)
