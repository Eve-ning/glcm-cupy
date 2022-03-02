import cupy as cp
import numpy as np
from cupyx.profiler import benchmark
from skimage.util import view_as_windows

glcm_kernel = cp.RawKernel(
    r"""
extern "C" {
    __global__ void glcm(
        const int* i,
        const int* j,
        int* glcm_ix,
        int* glcm,
        const int maxValue,
        const int noOfValues
    ) 
    {
    
        cudaMalloc();
        long int tid = blockDim.x * blockIdx.x + threadIdx.x;

        // Prevent excess threads from accessing OOB memory
        if (tid >= noOfValues) return;

        int row = i[tid];
        int col = j[tid];

        long int ix = col + row * maxValue + tid * maxValue * maxValue;

        glcm_ix[tid] = ix;
        glcm[ix] = 1;
        
        __syncthreads();
        
        printf( "Hello, World!\n" );
        /**
        long int ixtop = col + row * maxValue;
        if (ix == ixtop) {
            for (int i = 0; i < noOfValues; i++){
                glcm[ixtop] += glcm[i]
            }
        };
        **/
    }

}
    """, "glcm"
)

max_value = 128
im_size = (200, 300)
window_radius = 4
window_diameter = window_radius * 2 + 1

im = np.random.randint(0, max_value, im_size, dtype=cp.int_)
windows_ij = cp.asarray(
    view_as_windows(im, (window_diameter, window_diameter)))
windows_ij = windows_ij.reshape((*windows_ij.shape[:-2], -1))
windows_i = windows_ij[:-1, :-1].reshape((-1, windows_ij.shape[-1]))
windows_j = windows_ij[1:, 1:].reshape((-1, windows_ij.shape[-1]))
number_of_values = window_diameter ** 2
i = windows_i[0]
j = windows_j[0]
glcm_ix = cp.zeros((max_value, max_value), dtype=cp.int_)
glcm = cp.zeros((number_of_values, max_value, max_value),
                dtype=cp.int_)
MAX_THREADS = 256

if number_of_values > MAX_THREADS:
    raise Exception("Radius too large")


def b(glcm):
    glcm_kernel(
        grid=(16,),
        block=(number_of_values // 16,),
        args=(i, j, glcm_ix, glcm, max_value, number_of_values)
    )
    glcm = glcm.sum(axis=0)


# %%
print(benchmark(b, args=(glcm,), n_repeat=1))
# %%
glcm = glcm.sum(axis=0)
# %%
glcm.sum()
