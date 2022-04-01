Binning
=======

.. code-block:: python

    >>> from glcm_cupy import GLCM
    >>> import numpy as np
    >>> from PIL import Image
    >>> ar = np.asarray(Image.open("image.jpg"))
    >>> g = GLCM(..., bin_from=256, bin_to=16).run(ar)

To reduce the time taken for GLCM, we shrink GLCM size by limiting the maximum value of the input.

E.g. an image of maximum value 255 will require a :math:`256\times 256` sized GLCM.
Shrinking it to 15 will drastically reduce to :math:`16\times 16`.

The arguments are thus ``bin_from=256`` and ``bin_to=16``.

Caution
-------

If an image has a max value of 255, we use ``bin_from=256``.
This is due to the 0 included in binning, and simplicity.

Recommendations
---------------

I recommend using ``bin_to<=16`` for testing purposes, then upscaling when you're ready to use higher bins.

.. code-block:: python

    >>> from glcm_cupy import GLCM, Direction
    >>> g = GLCM(bin_from=256, bin_to=16)