Batch Processing
================

.. _batch_processing:

*Note: It's not faster than a loop on ``GLCMCross``*

If you have multiple images of the **SAME SIZE**, you may opt to run them concurrently.

This is slightly faster on ``GLCM``, it will start on the next image while the current is still being processed.

.. code-block:: python

    >>> from glcm_cupy import GLCM
    >>> import cv2
    >>> ar0 = cv2.imread("image_0.jpg")
    >>> ar1 = cv2.imread("image_1.jpg")
    >>> ar0.shape
    (1080, 1920, 3)
    >>> # This is why you need them to be the same size
    >>> g = GLCM(np.stack([ar0, ar1])).run(ar)
    >>> g.shape
    (2, 1074, 1914, 3, 6)
    >>> g0, g1 = g[0], g[1]

The dimensions here are ``(batch, height, width, channel, features)``.

So, to retrieve the 1st image's glcm, `g[0]`

GLCMCross
---------

``GLCMCross`` does not benefit in speed, however it may be simpler in syntax.

.. code-block:: python

    >>> g = GLCMCross(np.stack([ar0, ar1])).run(ar)
    >>> g.shape
    (2, 1074, 1914, 3, 6)
    >>> g0, g1 = g[0], g[1]

Behavior of the ``ix_combos`` specified does not change.

If it's blank, it'll generate all combinations for each image.
