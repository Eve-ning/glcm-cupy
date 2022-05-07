GLCM Features
=============

.. code-block:: python

    >>> from glcm_cupy import GLCM, CONTRAST, CORRELATION
    >>> import numpy as np
    >>> from PIL import Image
    >>> ar = np.asarray(Image.open("image.jpg"))
    >>> g = GLCM(...).run(ar)
    >>> print(g[..., CONTRAST])
    >>> print(g[..., CORRELATION])

I recommend referencing `this tutorial <https://prism.ucalgary.ca/handle/1880/51900>`_ for knowledge gaps here

In total, we have 6 features.

- Homogeneity
- ASM
- Contrast
- Correlation
- Mean
- Variance

.. math::

    \text{Homogeneity} = \sum_{i,j=0}^{N-1}\frac{P_{i,j}}{1+(i-j)^2}\\
    \text{Contrast} = \sum_{i,j=0}^{N-1}P_{i,j}(i-j)^2\\
    \text{Angular Second Moment (ASM)} = \sum_{i,j=0}^{N-1}P_{i,j}^2\\
    \text{GLCM Mean, } \mu = \sum_{i,j=0}^{N-1}i * P_{i,j}\\
    \text{GLCM Variance, } \sigma^2 = \sum_{i,j=0}^{N-1}P_{i,j}(i - \mu_i)^2\\
    \text{Correlation} = \frac{(i - \mu_i)(j - \mu_j)}{\sqrt{\sigma_i^2\sigma_j^2}}\\

We implemented these few as they are the most orthogonal (according to the tutorial).
However, feel free to suggest any more in the GitHub Issue page.
