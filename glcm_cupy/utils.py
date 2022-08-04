from __future__ import annotations

from typing import Tuple

from cupy.lib.stride_tricks import as_strided

from glcm_cupy.conf import *


def normalize_features(ar: ndarray, bin_to: int) -> ndarray:
    """ This scales the glcm features to [0, 1] """
    ar[..., Features.CONTRAST] /= (bin_to - 1) ** 2
    ar[..., Features.MEAN] /= (bin_to - 1)
    ar[..., Features.VARIANCE] /= (bin_to - 1) ** 2
    ar[..., Features.CORRELATION] += 1
    ar[..., Features.CORRELATION] /= 2
    ar[..., Features.DISSIMILARITY] /= (bin_to - 1)
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


def binner(im: cp.ndarray, bin_from: int, bin_to: int) -> cp.ndarray:
    """ Bins an image from a certain bin to another

    Args:
        im: Image as np.ndarray or cp.ndarray
        bin_from: From the Bin of input image
        bin_to: To the Bin of output image

    Returns:
        Binned Image as np.uint8 or cp.uint8

    """
    # Convert to compatible types
    return (im.astype(cp.float32) / bin_from * bin_to).astype(cp.uint8)


def nan_to_num(im: ndarray, nan: int) -> ndarray:
    """ Converts nan to another value

    Args:
        im: Image as np.ndarray or cp.ndarray
        nan: Target value

    """
    if isinstance(im, cp.ndarray):
        return cp.nan_to_num(im, copy=False, nan=nan)
    return np.nan_to_num(im, copy=False, nan=nan)


def view_as_windows_cp(arr_in: cp.ndarray, window_shape, step=1):
    """ Rolling window view of the input n-dimensional array.

    Notes:
        Adapted from ``skimage.util import view_as_windows``
    """
    ndim = arr_in.ndim
    arr_shape = cp.array(arr_in.shape)
    window_shape = cp.array(window_shape, dtype=arr_shape.dtype)

    if step < 1:
        raise ValueError("`step` must be >= 1")
    step = (step,) * ndim

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = cp.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (((cp.array(arr_in.shape) - cp.array(window_shape))
                          // cp.array(step)) + 1)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=tuple(map(int, new_shape)),
                         strides=tuple(map(int, strides)))
    return arr_out
