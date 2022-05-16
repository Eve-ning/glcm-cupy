from __future__ import annotations

from typing import Tuple

import numpy as np

from glcm_cupy.conf import *


def normalize_features(ar: np.ndarray, bin_to: int):
    """ This scales the glcm features to [0, 1] """
    ar[..., CONTRAST] /= (bin_to - 1) ** 2
    ar[..., MEAN] /= (bin_to - 1)
    ar[..., VAR] /= (bin_to - 1) ** 2
    ar[..., CORRELATION] += 1
    ar[..., CORRELATION] /= 2
    return ar


def calc_grid_size(
    window_count: int,
    glcm_size: int,
    thread_per_block: int = MAX_THREADS
) -> Tuple[int, int]:
    """ Calculates the min. required grid size for CUDA """

    # Blocks to support features
    blocks_req_glcm_features = \
        window_count * glcm_size * glcm_size / thread_per_block

    # Blocks to support glcm populating
    blocks_req_glcm_populate = glcm_size * window_count

    # Take the maximum
    blocks_req = max(blocks_req_glcm_features, blocks_req_glcm_populate)

    # We split it to 2 dims: A -> B x B
    b = int(blocks_req ** 0.5) + 1
    return b, b


def binner(im: np.ndarray, bin_from: int, bin_to: int) -> np.ndarray:
    """ Bins an image from a certain bin to another

    Args:
        im: Image as np.ndarray
        bin_from: From the Bin of input image
        bin_to: To the Bin of output image

    Returns:
        Binned Image as np.uint8

    """
    return (im.astype(np.float32) / bin_from * bin_to).astype(np.uint8)
