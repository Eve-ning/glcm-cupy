# GLCM Bin 5 on CuPy

This is a CuPy reimplementation of my `glcmbin5` [**on my other repository**](https://github.com/Eve-ning/glcmbin5).

This directly utilizes CUDA to speed up the processing of GLCM.

# Installation

First, you need to install this

```shell
pip install glcm-cupy
```

Then, you need **CuPy**.
You need to [install CuPy manually](https://docs.cupy.dev/en/stable/install.html), 
as it's dependent on the version of CUDA you have.

I recommend using `conda-forge` as it worked for me :)

```shell
conda install -c conda-forge cupy cudatoolkit=__._
```

E.g. my CUDA is `11.6`, thus

```shell
conda install -c conda-forge cupy cudatoolkit=11.6
```

# Usage

The usage is simple:
```py
from glcm_cupy import GLCM
ar: np.ndarray = ...
g = GLCM(...).from_3dimage(ar)
g = GLCM(...).from_2dimage(ar)
```

```py
import numpy as np
from glcm_cupy import GLCM
from PIL import Image

ar = np.asarray(Image.open("image.jpg"))
g = GLCM(bin_from=256, bin_to=16).from_3dimage(ar)
```


## CUDA Notes

### Why is the kernel split into 4?

The kernel is split into 4 sections

1) GLCM Creation
2) Features (ASM, Contrast, Homogeneity, GLCM Mean I, GLCM Mean J)
3) Features (GLCM Variance I, GLCM Variance J)
4) Features (GLCM Correlation)

The reason why it's split is due to (2) being reliant on (1), and (3) on (2), ... .

There are some other solutions tried

1) `__syncthreads()` will not work as we require to sync all blocks.
    1) We can't put all calculations in a block due to the thread limit of 512, 1024, 2048.
    2) We require 256 * 256 threads minimum to support a GLCM of max value 255.
2) **Cooperative Groups** imposes a 24 block limit.

Thus, the best solution is to split the kernel.

### Atomic Add

Threads cannot write to a single pointer in parallel, information will be overwritten and lost. This is the **Race
Condition**.

In order to avoid this, we use [**Atomic
Functions**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions).

> ... it is guaranteed to be performed without interference from other threads

### Custom Atomic Add

Currently `atomicAdd()` doesn't have the signature to support `uint8` or `unsigned char`. We get this implementation
from this [**StackOverflow
Answer**](https://stackoverflow.com/questions/5447570/cuda-atomic-operations-on-unsigned-chars)

