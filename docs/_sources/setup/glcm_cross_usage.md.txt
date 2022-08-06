# Usage (GLCM Cross)

Given that `ar` is an `np.ndarray`.

**OOP** GLCM Cross Transform

```pycon
>>> from glcm_cupy import GLCMCross
>>> g = GLCMCross(...).run(ar)
```

**Functional** GLCM Cross Transform

```pycon
>>> from glcm_cupy import glcm_cross
>>> g = glcm_cross(ar, ...)
```

## I/O

For a `np.ndarray` or `cp.ndarray` input, the algorithm will output a
`np.ndarray` or `cp.ndarray` respectively.

### Input Array Requirements

- Shape: {math}`B\times H\times W\times C` or {math}`H\times W\times C`
- Non-negative, integer array
- Must be large enough for `radius`
- Must have at least 2 channels to cross

```{math}

B: \text{Batches}\\
H: \text{Height}\\
W: \text{Width}\\
C: \text{Channel}
```

### Output Array

- Shape: {math}`B\times H^*\times W^*\times C_{combo}\times F` or {math}`H\times W\times C_{combo}\times F`
- 0 to 1 if `normalize_features`, else it depends on feature.

```{math}

H^*: H - \text{Radius} \times 2\\
W^*: W - \text{Radius} \times 2\\
C_{combo}: {C \choose 2} = \text{Number of Pair Combinations}\\
F: \text{Features}
```

### Combinations

The pair combination order is dependent on `itertools.combinations`.

For example, for an image with 4 channels:

```pycon
>>> from itertools import combinations
>>> n_channels = 4
>>> pair_combinations = combinations(range(n_channels), 2)
>>> print(pair_combinations)
[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
```

We see that the 1st channel crosses **Channel 0** and **Channel 1**.

## Arguments

| Argument                                   | Description                             | Default                                |
|--------------------------------------------|-----------------------------------------|----------------------------------------|
| [features](features)                       | List of `GLCM.Features` for GLCM to use | `(Features.HOMOGENEITY, Features....)` |
| [ix_combos](ix_combos)                     | List of pair combinations to use        | `None` (All combinations)              |
| [radius](radius)                           | Radius of GLCM Windows                  | `2`                                    |
| [bin_from](bin_from)                       | Maximum Value + 1 of the array.         | `256`                                  |
| [bin_to](bin_to)                           | Maximum Value + 1 of the array          | `256`                                  |
| [normalized_features](normalized_features) | Whether to scale features to `[0, 1]`   | `True`                                 |
| verbose                                    | Whether `tqdm` outputs progress         | `True`                                 |
| max_partition_size[^*]                     | No. of windows parsed per GLCM Matrix   | `10000`                                |
| max_threads[^*]                            | No. of threads per CUDA block           | `512`                                  |

[^*]: Recommend to not change.

```{seealso}

Learn how to use `glcm-cupy` from the examples in the sidebar on the left!
```

