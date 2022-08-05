""" Example for Image Transformation with Padding """
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
# %%
from mpl_toolkits.axes_grid1 import ImageGrid

ar = imread(Path(__file__).parents[1] / "data/image.jpg")[100:200, 100:200]

# %%
# Observe the size of the transformed image
from glcm_cupy import glcm

radius = 3
step_size = 2
g = glcm(ar, bin_from=256, bin_to=16, radius=radius, step_size=step_size)
print(ar.shape, g.shape)
# (100, 100, 3) (90, 90, 3, 7)

# %%
# Add padding to border to avoid reduction of size
# The formula is: radius + step_size.
# Note: Direction specified is independent of this formula.
padding = radius + step_size
ar_pad = np.pad(ar,
                pad_width=((padding,), (padding,), (0,)),
                constant_values=np.nan)
g = glcm(ar_pad, bin_from=256, bin_to=16, radius=radius, step_size=step_size)
print(ar.shape, g.shape)
# (100, 100, 3) (100, 100, 3, 7)
# %%

# Plot in a grid
fig = plt.figure(figsize=(12, 8))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(3, 3),
                 axes_pad=0.4)

for ax, f_ix, title in zip(grid, range(g.shape[-1]),
                           ("HOMOGENEITY", "CONTRAST", "ASM", "MEAN",
                            "VARIANCE", "CORRELATION", "DISSIMILARITY"), ):
    ax.imshow(g[..., f_ix] ** (1 / 3))
    ax.set_title(title)
fig.suptitle('Padded GLCM Features (Cube Rooted for visibility)')
fig.show()
