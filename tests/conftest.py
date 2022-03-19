import os

import pytest


@pytest.fixture(autouse=True)
def set_env():
    os.environ['CUPY_EXPERIMENTAL_SLICE_COPY'] = '1'
