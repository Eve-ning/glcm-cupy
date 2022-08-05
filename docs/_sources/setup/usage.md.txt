# Usage (GLCM)

Given that `ar` is an `np.ndarray`.

**OOP** GLCM Transform

```pycon
>>> from glcm_cupy import GLCM
>>> g = GLCM(...).run(ar)
```

**Functional** GLCM Transform

```pycon
>>> from glcm_cupy import glcm
>>> g = glcm(ar, ...)
```

## I/O

For a `np.ndarray` or `cp.ndarray` input, the algorithm will output a
`np.ndarray` or `cp.ndarray` respectively.

### Input Array Requirements

- Shape: {math}`B\times H\times W\times C` or {math}`H\times W\times C`
- Non-negative, integer array
- Must be large enough for `radius` and `step_size`

```{math}

B: \text{Batches}\\
H: \text{Height}\\
W: \text{Width}\\
C: \text{Channel}
```

### Output Array

- Shape: {math}`B\times H^*\times W^*\times C\times F` or {math}`H\times W\times C\times F`
- 0 to 1 if `normalize_features`, else it depends on feature.

```{math}

H^*: H - (\text{Step Size} + \text{Radius}) \times 2\\
W^*: W - (\text{Step Size} + \text{Radius}) \times 2\\
F: \text{Features}
```

## Arguments

| Argument              | Description                              | Default                                |
|-----------------------|------------------------------------------|----------------------------------------|
| `directions`          | List of `GLCM.Direction` for GLCM to use | `(Direction.EAST, Direction....)`      |
| `features`            | List of `GLCM.Features` for GLCM to use  | `(Features.HOMOGENEITY, Features....)` |
| `step_size`           | Distance between GLCM Windows            | `1`                                    |
| `radius`              | Radius of GLCM Windows                   | `2`                                    |
| `bin_from`            | Maximum Value + 1 of the array.          | `256`                                  |
| `bin_to`              | Maximum Value + 1 of the array           | `256`                                  |
| `normalized_features` | Whether to scale features to `[0, 1]`    | `True`                                 |
| `verbose`             | Whether `tqdm` outputs progress          | `True`                                 |
| *`max_partition_size` | No. of windows parsed per GLCM Matrix    | `10000`                                |
| *`max_threads`        | No. of threads per CUDA block            | `512`                                  |

```{note}
\* Recommend to not change
```

```{seealso}

- {ref}`Selectively Generate a GLCM Feature <select_feature>`:
- {ref}`Get a GLCM Feature <get_feature>`:
- {ref}`Process many same-sized images at once <batch_processing>`:
```

