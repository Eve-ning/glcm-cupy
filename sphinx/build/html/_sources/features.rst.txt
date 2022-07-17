GLCM Feature Indexing
=====================

.. _select_feature:

Selecting Features
------------------
To reduce parsing time, you can selectively parse specific features.

This means that other features returned will be zeroed (or 0.5 if normalized correlation).

Note that the other of ``features`` does not determine the order of ``g``'s channel.

.. code-block:: python

    >>> from glcm_cupy import GLCM, CONTRAST, CORRELATION
    >>> import cv2
    >>> ar = cv2.imread("image.jpg")
    >>> g = GLCM(..., features=(CONTRAST, CORRELATION)).run(ar)

.. _get_feature:

Getting Features
----------------
To retrieve the features, use the following syntax

.. code-block:: python

    >>> from glcm_cupy import GLCM, CONTRAST, CORRELATION
    >>> print(g[..., CONTRAST])
    >>> print(g[..., CORRELATION])

Feature Theory
--------------

I recommend referencing `this tutorial <https://prism.ucalgary.ca/handle/1880/51900>`_ for knowledge gaps here

In total, we have 6 features.

.. math::

    \text{Homogeneity} = \sum_{i,j=0}^{N-1}\frac{P_{i,j}}{1+(i-j)^2}\\
    \text{Contrast} = \sum_{i,j=0}^{N-1}P_{i,j}(i-j)^2\\
    \text{Angular Second Moment (ASM)} = \sum_{i,j=0}^{N-1}P_{i,j}^2\\
    \text{GLCM Mean, } \mu = \sum_{i,j=0}^{N-1}i * P_{i,j}\\
    \text{GLCM Variance, } \sigma^2 = \sum_{i,j=0}^{N-1}P_{i,j}(i - \mu_i)^2\\
    \text{Correlation} = \frac{(i - \mu_i)(j - \mu_j)}{\sqrt{\sigma_i^2\sigma_j^2}}\\
    \text{Dissimilarity} = \sum_{i,j=0}^{N-1}P_{i,j} * \left\lvert i - j \right\rvert\\

We implemented these few as they are the most orthogonal (according to the tutorial).
However, feel free to suggest any more in the GitHub Issue page.
