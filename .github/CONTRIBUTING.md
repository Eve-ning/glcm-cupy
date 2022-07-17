# Contributing to GLCM CuPy

Thanks for choosing to contribute to this project!

## Before you submit a PR

I heavily recommend [creating an issue](https://github.com/Eve-ning/glcm-cupy/issues).

So I can determine if it's worthwhile to proceed, else risk your hardwork being rejected (I don't want that too!).

-----

## Submitting an Issue

Any issue is fine:
- Bug Report
- Feature Request
- Questions

-----

## Submitting a PR

Given the "OK", or if you're confident, proceed in creating a PR:

- [Any Pull Requests must be _locally_ tested.](local-testing)
- [Documentation must be updated.](updating-documentation)

### Local Testing

Testing locally is simple: run `pytest tests/`

❔[(Help needed) Unfortunately, **GitHub Actions** don't support CUDA, ... or does it?](https://github.com/Eve-ning/glcm-cupy/pull/24)

### GLCM Testing

#### Cross Checking with Python

We have 2 versions of GLCM:
1) Pure Python
    1) `glcm_cupy/glcm_py_base.py`
    2) `glcm_cupy/glcm/glcm_py.py`
    3) `glcm_cupy/cross/glcm_cross_py.py`
3) CUDA 
    1) `glcm_cupy/glcm_base.py`. Uses: `glcm_cupy/kernel.py`
    2) `glcm_cupy/glcm/glcm.py`
    3) `glcm_cupy/cross/glcm_cross.py`

To assert expected behavior, the **Pure Python** implementation is essential to cross check against **CUDA**.

Thus, if you add a feature, **both GLCMs must be updated.** Ensure (1.) is correct and both should be correct.

#### Consistency Checking

Upon different runs of CUDA, values differ by small amounts

To assert that it's largely consistent, a cached `.npy` file in `integration_tests` is used for comparison.

Furthermore, it asserts consistency between NumPy and CuPy array implementations

### Updating Documentation

I use **Sphinx** for documentation.

- ❌ Do **NOT** update documentation directly in `docs/`, it's the compilation of files in `sphinx/src`
- ✔️ Update Documentation in `sphinx/src`. Ping me if you need help generating the documentation, don't worry, it's trivial.
