GLCM Implemented in CuPy
========================

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    features
    binning
    direction
    radius_step_size
    acknowledgements

Quick Start
===========

You need Python >= 3.8

First, you need to install this

.. code-block:: shell

   pip install glcm-cupy

Then, you need **CuPy**.
You need to `install CuPy manually <https://docs.cupy.dev/en/stable/install.html>`_
as it's dependent on the version of CUDA you have.

I recommend using ``conda-forge`` as it worked for me :)

For CUDA ``11.6``, we use

.. code-block:: shell

   conda install -c conda-forge cupy cudatoolkit=11.6

Replace the version you have on the arg.

.. code-block:: shell

   conda install -c conda-forge cupy cudatoolkit=__._
                                                 ^^^^
                                             CUDA Version

Usage
=====

Use GLCM on an image like so.

.. code-block:: python

    >>> from glcm_cupy import GLCM
    >>> import numpy as np
    >>> from PIL import Image
    >>> ar = np.asarray(Image.open("image.jpg"))
    >>> ar.shape
    (1080, 1920, 3)
    >>> g = GLCM(...).run(ar)
    >>> g.shape
    (1074, 1914, 3, 8)

The last dimension of `g` is the GLCM Features.

To retrieve a GLCM Feature:

.. code-block:: python

    >>> from glcm_cupy import CONTRAST
    >>> g[..., CONTRAST].shape

You may also consider simply `glcm` if you're not reusing `GLCM()`

.. code-block:: python

    >>> from glcm_cupy import glcm
    >>> g = glcm(ar, ...)

Read More
=========

View the articles on the left side-bar to learn more!

- `More about GLCM <https://prism.ucalgary.ca/handle/1880/51900>`_