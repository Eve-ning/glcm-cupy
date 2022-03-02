import time

import cupy as cp

from cupyx.profiler import benchmark

MatAdd = cp.RawKernel(
    r"""
        // Kernel definition
        __global__ void MatAdd(float A[10][10], float B[10][10],
                               float C[10][10])
        {
            int i = threadIdx.x;
            int j = threadIdx.y;
            C[i][j] = A[i][j] + B[i][j];
        }
    """, "MatAdd"
)


a = cp.ones((10,10), dtype=cp.float32)
b = cp.ones((10,10), dtype=cp.float32)
c = cp.zeros((10,10), dtype=cp.float32)

MatAdd((1,), (32,), (a,b,c))
