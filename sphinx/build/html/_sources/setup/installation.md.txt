# Installation

**Python >= 3.7**

First, you need to install this

```shell
pip install glcm-cupy
```

Then, you need to install **CuPy** version corresponding to your CUDA version

I recommend using `conda-forge` as it worked for me :)

```shell
conda install -c conda-forge cupy cudatoolkit=<your_CUDA_version>
```

E.g:
For CUDA `11.6`,

```shell
conda install -c conda-forge cupy cudatoolkit=11.6
```

To install **CuPy** manually, see [this page](https://docs.cupy.dev/en/stable/install.html)

```{note}

This supports **RAPIDS** `cucim`, automatically enabled if installed.
[RAPIDS Installation Guide](https://rapids.ai/start.html#requirements)
```
