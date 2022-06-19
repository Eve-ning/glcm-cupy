from pathlib import Path

import typing
import cupy as cp
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.absolute()

MAX_VALUE_SUPPORTED = 256
NO_OF_VALUES_SUPPORTED = 256 ** 2

MAX_THREADS = 512  # Min. Maximum supported threads.

# For a 1000 x 256 x 256 GLCM, you need ~ 262MB of memory.
# Each cell is 4 bytes (float32)
MAX_PARTITION_SIZE = 1000

NO_OF_FEATURES = 6

HOMOGENEITY = 0
CONTRAST = 1
ASM = 2
MEAN = 3
VAR = 4
CORRELATION = 5

ndarray = typing.TypeVar('ndarray', np.ndarray, cp.ndarray)
