NaN Handling
============

NaN Values, ``np.nan``, ``cp.nan``, are ignored. This means, they are **skipped** when populating GLCM.

This creates a **Partial GLCM**

Take for example a 4 x 4 image with 3 x 3 windows

.. code-block::

    # # # #  Window  # # #     1 2 3  Pairs  (#, 1), (#, 2), (#, 3) Filter (1, 6)
    # 1 2 3   --->   # 1 2  &  5 6 7   --->  (#, 5), (1, 6), (2, 7)  --->  (2, 7)
    4 5 6 7          4 5 6     9 # #         (4, 9), (5, #), (6, #)        (4, 9)
    8 9 # #

Thus, we will have a probability GLCM of 3 1/3 cells.

Padding
-------

Thus, using the above, you may consider padding your image with NaNs. Thus, creating **Partial GLCMs** for features
at the borders.

