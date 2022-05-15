Radius & Step Size
==================

.. code-block:: python

    >>> from glcm_cupy import GLCM
    >>> import numpy as np
    >>> from PIL import Image
    >>> ar = np.asarray(Image.open("image.jpg"))
    >>> g = GLCM(radius=1, step_size=1).run(ar)

Radius
------
The radius defines the window radius for each GLCM window.

If the radius is 1, we have diameter of 3 as it includes the center pixel

Step Size
---------
The step size defines the distance between each window.

If it's diagonal, it treats a diagonal step as 1. It's not the euclidean distance.

Checking Suitability
--------------------

If the image is too small for the radius & step size it will be rejected.

You may check the resulting shape ahead of time by

.. code-block:: python

    >>> from glcm_cupy import GLCM, CONTRAST, CORRELATION, glcm_shape
    >>> import numpy as np
    >>> from PIL import Image
    >>> ar = np.asarray(Image.open("image.jpg"))
    >>> shape_after = glcm_shape(ar, step_size=..., radius=...)

If ``shape_after`` has any values ``<=0``, you know that it'll fail.

This is the signature of that helper function

.. code-block:: python

    def glcm_shape(im_shape: Tuple, step_size: int, radius: int):
        ...