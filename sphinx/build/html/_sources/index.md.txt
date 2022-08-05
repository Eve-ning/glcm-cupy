# GLCM Implemented in CuPy

```{toctree}
---
maxdepth: 1
---

GLCM Features <features>
GLCM Directions <direction>
Binning <binning>
NaN Handling <nan_handling>
Radius & Step Size <radius_step_size>
Cross GLCM <glcm_cross>
Batch Processing <batch_processing>
```

```{toctree}
---
maxdepth: 1
hidden:
---

Acknowledgements <acknowledgements>
Change Log <changelog>
```

## Quick Start

**Python >= 3.7**

First, you need to install this

```shell
pip install glcm-cupy
```

Then, you need to install **CuPy** version corresponding to your CUDA version

I recommend using `conda-forge` as it worked for me :)

```shell
conda install -c conda-forge cupy cudatoolkit=<your_CUDA_version>
```

E.g:
For CUDA `11.6`,

```shell
conda install -c conda-forge cupy cudatoolkit=11.6
```

To install **CuPy** manually, see [this page](https://docs.cupy.dev/en/stable/install.html)

### Optional Installation

This supports **RAPIDS** `cucim`.

[RAPIDS Installation Guide](https://rapids.ai/start.html#requirements)

*It's automatically enabled if installed.*

## Usage

Use GLCM on an image like so.

```pycon
>>> from glcm_cupy import GLCM
>>> import cv2
>>> ar = cv2.imread("image.jpg")
>>> ar.shape
(1080, 1920, 3)
>>> g = GLCM(...).run(ar)
>>> g.shape
(1074, 1914, 3, 6)
```

Last dimension of ``g`` is the GLCM Features.

To {ref}`selectively generate a GLCM Feature <select_feature>`:

```pycon
>>> from glcm_cupy import CONTRAST, CORRELATION
>>> g = GLCM(..., features=(CONTRAST, CORRELATION)).run(ar)
```

To {ref}`get a GLCM Feature <get_feature>`:

```pycon
>>> from glcm_cupy import CONTRAST
>>> g[..., CONTRAST].shape
```

Consider ``glcm`` if not reusing ``GLCM()``

```pycon
>>> from glcm_cupy import glcm
>>> g = glcm(ar, ...)
```

{ref}`Process many same-sized images at once <batch_processing>`:

```pycon
>>> g = GLCM(...).run(np.stack([ar0, ar1, ar2]))
>>> g0, g1, g2 = g
```

## Read More

View the articles on the left side-bar to learn more!

- [GLCM Texture: A Tutorial v. 3.0 March 2017](https://prism.ucalgary.ca/handle/1880/51900)
