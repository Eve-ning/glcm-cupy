import time

import cupy as cp
from cupyx.profiler import benchmark

glcm_kernel = cp.RawKernel(
    r"""
extern "C" {

    __global__ void glcm(
        const int* i,
        const int* j,
        int* glcm_ix,
        bool* glcm,
        const int maxValue,
        const int noOfValues
    ) 
    {
        long int tid = blockDim.x * blockIdx.x + threadIdx.x;
        
        // Prevent excess threads from accessing OOB memory
        if (tid >= noOfValues) return;
        
        int row = i[tid];
        int col = j[tid];
        
        long int ix = col + row * maxValue + tid * maxValue * maxValue;
        
        glcm_ix[tid] = ix;
        glcm[ix] = true;
    }

}
    """, "glcm"
)
radius = 3
diameter = radius * 2 + 1
max_value = 10
number_of_values = diameter ** 2

i = cp.random.randint(0, max_value - 1, (diameter, diameter),
                      dtype=cp.int_).flatten()
j = cp.random.randint(0, max_value - 1, (diameter, diameter),
                      dtype=cp.int_).flatten()
glcm_ix = cp.zeros((max_value, max_value), dtype=cp.int_)
glcm = cp.zeros((number_of_values, max_value, max_value),
                dtype=cp.bool_)
MAX_THREADS = 256

if number_of_values > MAX_THREADS:
    raise Exception("Radius too large")
glcm_kernel(
    grid=(20,),
    block=(diameter ** 2,),
    args=(i, j, glcm_ix, glcm, max_value, number_of_values)
)

# %%
glcm.sum()

# %%

# %%
from skimage.util import view_as_windows
import numpy as np
import cupy as cp

max_value = 256
im_size = (200, 300)
window_radius = 1
window_diameter = window_radius * 2 + 1

im = np.random.randint(0, max_value, im_size, dtype=cp.int_)
ws = view_as_windows(im, (window_diameter, window_diameter)) \
    .reshape((-1, window_diameter, window_diameter))
# %%


