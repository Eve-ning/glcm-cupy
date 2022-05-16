Cross GLCM
==========

Cross GLCM is when I & J for GLCM is taken from separate images.

Thus, you're yielding statistical relationships between images

Usage
-----

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

Last dimension of ``g`` is the GLCM Features.

To retrieve a GLCM Feature:

.. code-block:: python

    >>> from glcm_cupy import CONTRAST
    >>> g[..., CONTRAST].shape

Consider ``glcm_cross`` if not reusing ``GLCMCross()``

.. code-block:: python

    >>> from glcm_cupy import glcm
    >>> g = glcm_cross(ar, ...)

Selecting and Retrieving Combinations
-------------------------------------

In each ``GLCMCross()`` instance, is a ``ix_combos`` property to specify combinations

In a 3 channel image, the following will only cross indices, 0 with 1, 1 with 2, **skipping** 0 with 2

.. code-block:: python

    >>> from glcm_cupy import glcm
    >>> g = GLCMCross(..., ix_combos=[[0, 1], [1, 2]]).run(ar)
    >>> g.shape
    (1080, 1920, 2, 6)

By default, ``ix_combos is None``, using all possible combinations.

After ``run(...)``, ``ix_combos`` will be populated with combinations used.

Thus, you can retrieve the default ordering of the 3rd dimension.

The combinations is in the order of ``itertools.combinations(range(CHANNELS), 2)``.

**Caveat:** When using ``glcm_cross``, you can't retrieve the generated order.
However, you may use ``itertools.combinations`` to indirectly yield it.
