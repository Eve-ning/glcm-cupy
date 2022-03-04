# GLCM Bin 5 on CuPy

This is a CuPy reimplementation of my `glcmbin5` [**on my other repository**](https://github.com/Eve-ning/glcmbin5).

This directly utilizes CUDA to speed up the processing of GLCM.

## Notes

### Why is the kernel split into 3?

The kernel is split into 3 sections
1) GLCM Creation & the Features (ASM, Contrast, Homogeneity, GLCM Mean I, GLCM Mean J)
2) Features (GLCM Variance I, GLCM Variance J)
3) Features (GLCM Correlation)

The reason why it's split is due to (2) being reliant on (1), and (3) on (2).

There are some other solutions tried

1) `__syncthreads()` will not work as we require to sync all blocks.
   1) We can't put all calculations in a block due to the thread limit of 512, 1024, 2048.
   2) We require 256 * 256 threads minimum to support a GLCM of max value 255. 
2) **Cooperative Groups** imposes a 24 block limit.

Thus, the best solution is to split the kernel.

### Atomic Add

Threads cannot write to a single pointer in parallel, information will be overwritten and lost.
This is the **Race Condition**.

In order to avoid this, we use [**Atomic Functions**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions).

> ... it is guaranteed to be performed without interference from other threads

### Custom Atomic Add

Currently `atomicAdd()` doesn't have the signature to support `uint8` or `unsigned char`. We get
this implementation from this [**StackOverflow Answer**](https://stackoverflow.com/questions/5447570/cuda-atomic-operations-on-unsigned-chars)

