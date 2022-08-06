# GLCM Cross Index Combinations

(ix_combos)=

By default, if `ix_combos` is None, all combinations are used.
You can override the combinations.

For example for 3 channels, if you only want to cross **Channel 0 & 1** then **Channel 1 & 2**,
then `ix_combos=[(0, 1), (1, 2)]`

```pycon
>>> from glcm_cupy import GLCMCross, Direction
>>> g = GLCMCross(ix_combos=[(0, 1), (1, 2)])
```

In a full example:

```pycon
>>> from glcm_cupy import GLCM, Direction
>>> import numpy as np
>>> from PIL import Image
>>> ar = np.asarray(Image.open("image.jpg"))
>>> g = GLCMCross(ix_combos=[(0, 1), (1, 2)])
```

```{note}

The order of `ix_combos` will impact the resulting array.
```

```{warning}

If `ix_combos=[(0, 1), (0, 1)]`, note that the computation will be repeated!
```