# Direction

```pycon
>>> from glcm_cupy import GLCM, Direction
>>> import numpy as np
>>> from PIL import Image
>>> ar = np.asarray(Image.open("image.jpg"))
>>> g = GLCM(directions=(Direction.SOUTH_WEST, Direction.SOUTH)).run(ar)
```

I recommend referencing [GLCM Texture: A Tutorial v. 3.0 March 2017](https://prism.ucalgary.ca/handle/1880/51900) for
knowledge gaps here

## Bi-Directionality

This algorithm uses bi-directional algorithm in the kernel. It populates GLCM for I, J and J, I.

> This bi-directionality performance cost is negligible as it simply adds the transposed GLCM.

## Directions

You can specify directions from 4 directions.

- East: ``Direction.EAST``
- South East: ``Direction.SOUTH_EAST``
- South: ``Direction.SOUTH``
- South West: ``Direction.SOUTH_WEST``

The other 4 is unavailable as it's covered by bi-directionality

We can thus specify them as such.

```pycon
>>> from glcm_cupy import GLCM, Direction
>>> g = GLCM(directions=(Direction.SOUTH_WEST, Direction.SOUTH))
```
