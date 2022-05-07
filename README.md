# GLCM Bin 5 on CuPy

This is a CuPy reimplementation of my `glcmbin5` [**on my other repository**](https://github.com/Eve-ning/glcmbin5).

This directly utilizes CUDA to speed up the processing of GLCM.

# Installation

**Python >= 3.7**

First, you need to install this

```shell
pip install glcm-cupy
```

Then, you need **CuPy**.
You need to [install CuPy manually](https://docs.cupy.dev/en/stable/install.html), 
as it's dependent on the version of CUDA you have.

I recommend using `conda-forge` as it worked for me :)

For CUDA `11.6`, we use
```shell
conda install -c conda-forge cupy cudatoolkit=11.6
```

Replace the version you have on the arg.

```shell
conda install -c conda-forge cupy cudatoolkit=__._
```

# Usage

The usage is simple:

```pycon
>>> from glcm_cupy import GLCM
>>> import numpy as np
>>> from PIL import Image
>>> ar = np.asarray(Image.open("image.jpg"))
>>> ar.shape
(1080, 1920, 3)
>>> g = GLCM(...).run(ar)
>>> g.shape
(1074, 1914, 3, 8)
```

The last dimension of `g` is the GLCM Features.

To retrieve a specific GLCM Feature:

```pycon
>>> from glcm_cupy import CONTRAST
>>> g[..., CONTRAST].shape
(1074, 1914, 3)
```

You may also consider simply `glcm` if you're not reusing `GLCM()`
```pycon
>>> from glcm_cupy import glcm
>>> g = glcm(ar, ...)
```

## **[Example: Processing an Image](examples/process_an_image/main.py)**

## Features

These are the features implemented.

- `HOMOGENEITY = 0`
- `CONTRAST = 1`
- `ASM = 2`
- `MEAN_I = 3`
- `MEAN_J = 4`
- `VAR_I = 5`
- `VAR_J = 6`
- `CORRELATION = 7`

Don't see one you need? Raise an issue, I'll (hopefully) add it.

## Radius & Step Size

- The radius defines the window radius for each GLCM window.
- The step size defines the distance between each window.
  - If it's diagonal, it treats a diagonal step as 1. It's not the euclidean distance.

## Binning

To reduce GLCM processing time, you can specify `bin_from` & `bin_to`.

This will bin the image from a range to another.

I highly recommend using this to reduce time taken before raising it.

E.g.

> I have an RGB image with a max value of 255.
> 
> I limit the max value to 31. This reduces the processing time.
> 
> `GLCM(..., bin_from=256, bin_to=32).run(ar)`

The lower the max value, the smaller the GLCM required. Thus allowing for
more GLCMs to run concurrently.

## Direction

By default we have the following directions to run GLCM on.

- East: `Direction.EAST`
- South East: `Direction.SOUTH_EAST`
- South: `Direction.SOUTH`
- South West: `Direction.SOUTH_WEST`

For each direction, the GLCM will be bi-directional.

We can specify only certain directions here.

```pycon
>>> from glcm_cupy import GLCM
>>> GLCM()
>>> g = GLCM(directions=(Direction.SOUTH_WEST, Direction.SOUTH))
```

The result of these directions will be averaged together.

# Notes

> Q: Why did my image shrink?
> 
> The image shrunk due to `step_size` & `radius`.
> 
> The amount of shrink per XY Dimension is
> `size - 2 * step_size - 2 * radius`

> Q: What's the difference between this and `glcmbin5`?
> 
> This is the faster one, and easier to use.
> I highly recommend avoiding `glcmbin5` as it has C++, which means you need to compile manually.
> 
> It's the first version of GLCM I made.

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

# Change Log

## 1.6

Dropped dependency on J variables as I & J are always the same

## 1.7

Fix issue with GLCM overflowing by making it `float32`
