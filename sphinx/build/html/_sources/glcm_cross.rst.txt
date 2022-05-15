Cross GLCM
==========

Cross GLCM is when I & J for GLCM is taken from separate images.

Thus, you're yielding statistical relationships between images

Usage
=====

Use Cross GLCM on an image like so.

.. code-block:: python

    >>> from glcm_cupy import GLCMCross
    >>> import numpy as np
    >>> from PIL import Image
    >>> ar = np.asarray(Image.open("image.jpg"))
    >>> ar.shape
    (1080, 1920, 3)
    >>> g = GLCMCross(...).run(ar)
    >>> g.shape
    (1080, 1920, 3, 6)

Last dimension of `g` is the GLCM Features.

To retrieve a GLCM Feature:

.. code-block:: python

    >>> from glcm_cupy import CONTRAST
    >>> g[..., CONTRAST].shape

Consider `glcm_cross` if not reusing `GLCMCross()`

.. code-block:: python

    >>> from glcm_cupy import glcm
    >>> g = glcm(ar, ...)
