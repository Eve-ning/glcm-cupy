# CUDA Notes

Some notes I took down while creating the kernel in `glcm_cupy.kernel`

## Why conditional modules?

A huge benefit to dynamically compiled CUDA is the option to remove unneeded code.

Thus, when [selecting features](select_feature) the compilations are different.

## Why is the kernel split into 4?

The kernel is split into 4 sections

1) GLCM Creation
2) Features (ASM, Contrast, Homogeneity, GLCM Mean I, GLCM Mean J, Dissimilarity)
3) Features (GLCM Variance I, GLCM Variance J)
4) Features (GLCM Correlation)

- (2) is dependent on (1)
- (3) is dependent on (2)
- (4) is dependent on (3)

It's not possible for a single kernel to sync all threads. Thus, they are separated.

There are some other solutions tried

1) `__syncthreads()` will not work as we require to sync all blocks.
    1) We can't put all calculations in a block due to the thread limit of 512, 1024, 2048.
    2) We require 256 * 256 threads minimum to support a GLCM of max value 255.
2) **Cooperative Groups** imposes a 24 block limit.

## Atomic Add

Threads cannot write to a single pointer in parallel, information will be overwritten and lost. This is the **Race
Condition**.

In order to avoid this, we use [**Atomic
Functions**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions).

> ... it is guaranteed to be performed without interference from other threads

