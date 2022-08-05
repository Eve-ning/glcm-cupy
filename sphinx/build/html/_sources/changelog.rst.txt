Changelog
=========

1.6
---
- Dropped dependency on J variables as I & J are always the same

1.7
---
- Fix issue with GLCM overflowing by making it float32

1.8
---
- Implement Cross GLCM

1.9
---

- Issues Resolved

    - `#9 <https://github.com/Eve-ning/glcm-cupy/issues/9>`_
    - `#11 <https://github.com/Eve-ning/glcm-cupy/issues/11>`_
    - `#13 <https://github.com/Eve-ning/glcm-cupy/issues/13>`_

1.10
----

- `Merged PR 18 <https://github.com/Eve-ning/glcm-cupy/pull/18>`_
    - Add support to CuPy as input
    - Add optional support for RAPIDS cuCIM
- `Merged PR 25 <https://github.com/Eve-ning/glcm-cupy/pull/25>`_
    - Implement Integration Test checking for stability of GLCM output
- Implemented GLCM Feature Selection to optimize out unnecessary GLCM Stages
- Fix issue with CuPy ndarray incompatible with tqdm
- Add Batch Processing for vanilla GLCM
    - ``GLCMCross`` doesn't perform faster with this. However is compatible.
- Implement Dissimilarity Feature
- Adjusts `PR 18 <https://github.com/Eve-ning/glcm-cupy/pull/18>`_ coerces inputs to CuPy if NumPy.
- `Implement NaN Handling by ignoring contribution to GLCM <https://github.com/Eve-ning/glcm-cupy/pull/36>`_
