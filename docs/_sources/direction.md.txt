# Direction

You can specify directions from 4 directions.

- East: ``Direction.EAST``
- South East: ``Direction.SOUTH_EAST``
- South: ``Direction.SOUTH``
- South West: ``Direction.SOUTH_WEST``

```pycon
>>> from glcm_cupy import GLCM, Direction
>>> g = GLCM(directions=(Direction.SOUTH_WEST, Direction.SOUTH))
```

In a full example:

```pycon
>>> from glcm_cupy import GLCM, Direction
>>> import numpy as np
>>> from PIL import Image
>>> ar = np.asarray(Image.open("image.jpg"))
>>> g = GLCM(directions=(Direction.SOUTH_WEST, Direction.SOUTH)).run(ar)
```

## Bi-Directionality

This algorithm uses bi-directional algorithm in the kernel. It populates GLCM for I, J and J, I.

```{note}

This bi-directionality performance cost is negligible as it simply adds the transposed GLCM.
```
