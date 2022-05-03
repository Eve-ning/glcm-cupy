from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()

MAX_VALUE_SUPPORTED = 256
NO_OF_VALUES_SUPPORTED = 256 ** 2

MAX_THREADS = 512  # Lowest Maximum supported threads.

NO_OF_FEATURES = 8

# For a 10000 x 256 x 256 GLCM, you need ~ 600MB of memory.
MAX_PARTITION_SIZE = 10000

HOMOGENEITY = 0
CONTRAST = 1
ASM = 2
MEAN_I = 3
VAR_I = 4
CORRELATION = 5