# Changelog

## 1.6

- Dropped dependency on J variables as I & J are always the same

## 1.7

- Fix issue with GLCM overflowing by making it float32

## 1.8

- [Merged PR 7](https://github.com/Eve-ning/glcm-cupy/pull/7)
  - Implement Cross GLCM

## 1.9

- [Merged PR 10](https://github.com/Eve-ning/glcm-cupy/pull/10)
  - [Resolved Issue #9](https://github.com/Eve-ning/glcm-cupy/issues/9)
  - Add bad `ndim` raise
  - Improve raise message on bad `ndim`
- [Merged PR 14](https://github.com/Eve-ning/glcm-cupy/pull/14)
  - [Resolved Issue #11](https://github.com/Eve-ning/glcm-cupy/issues/11)
  - Allow `tqdm` to be silenced
- [Merged PR 15](https://github.com/Eve-ning/glcm-cupy/pull/15)
  - [Resolved Issue #13](https://github.com/Eve-ning/glcm-cupy/issues/13)
  - `normalize_features` -> `normalized_features` 
  - Fix `glcm` and `glcm_cross` unexpected arg order behaviour
  - Remove `test__from_windows()` as it's redundant
  - Fix `test_from_2d_image` failing due to missing 3rd dimension
  - Remove `test_image_tiff` as it's redundant
  - Fix reference `GLCM._binner` to `binner` in `utils`


## 1.10

- [Merged PR 18](https://github.com/Eve-ning/glcm-cupy/pull/18)
    - Add support to CuPy as input
    - Add optional support for RAPIDS cuCIM
- [Merged PR 25](https://github.com/Eve-ning/glcm-cupy/pull/25)
    - Implement Integration Test checking for stability of GLCM output
- [Merged PR 27](https://github.com/Eve-ning/glcm-cupy/pull/27)
  - Implemented GLCM Feature Selection to optimize out unnecessary GLCM Stages
- [Merged PR 29](https://github.com/Eve-ning/glcm-cupy/pull/30)
  - Fix issue with CuPy ndarray incompatible with tqdm
- [Merged PR 30](https://github.com/Eve-ning/glcm-cupy/pull/30)
  - Add Batch Processing for vanilla GLCM
      - ``GLCMCross`` doesn't perform faster with this.
- [Merged PR 32](https://github.com/Eve-ning/glcm-cupy/pull/32)
  - Author: [Julio Faracco](https://github.com/jcfaracco) 
  - Implement Dissimilarity Feature
- [Merged PR 35](https://github.com/Eve-ning/glcm-cupy/pull/35)
  - Adjusts [PR 18](https://github.com/Eve-ning/glcm-cupy/pull/18) coerces inputs to CuPy if NumPy.
  - Reduces duplicate code for handling NumPy and CuPy arrays conditionally.
- [Merged PR 36](https://github.com/Eve-ning/glcm-cupy/pull/36)
  - Implement NaN Handling by ignoring contribution to GLCM 
