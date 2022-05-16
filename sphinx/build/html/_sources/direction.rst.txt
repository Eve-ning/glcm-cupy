Direction
=========

.. code-block:: python

    >>> from glcm_cupy import GLCM, Direction
    >>> import numpy as np
    >>> from PIL import Image
    >>> ar = np.asarray(Image.open("image.jpg"))
    >>> g = GLCM(directions=(Direction.SOUTH_WEST, Direction.SOUTH)).run(ar)

I recommend referencing `this tutorial <https://prism.ucalgary.ca/handle/1880/51900>`_ for knowledge gaps here

Bi-Directionality
-----------------

This algorithm will use a bi-directional algorithm in the kernel.

That means, it'll run GLCM for the I, J and J, I. This bi-directionality performance cost is negligible.

This is achieved simply by adding the reflected GLCM.

Directions
----------

On top of Bi-directionality, you may specify directions from 4 directions.

- East: ``Direction.EAST``
- South East: ``Direction.SOUTH_EAST``
- South: ``Direction.SOUTH``
- South West: ``Direction.SOUTH_WEST``

The other 4 is not available as it's covered by their bi-directionality

We can thus specify them as such.

.. code-block:: python

    >>> from glcm_cupy import GLCM, Direction
    >>> g = GLCM(directions=(Direction.SOUTH_WEST, Direction.SOUTH))

