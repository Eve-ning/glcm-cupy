""" Example for Image Transformation with holes """
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from mpl_toolkits.axes_grid1 import ImageGrid

ar = imread(Path(__file__).parents[1] / "data/image.jpg")[100:200, 100:200]

ar = ar.astype(np.float64)
# Create a gap/hole in the image
ar[40:-40, 40:-40] = np.nan
plt.imshow(ar / 256)
plt.title("Image with NaN Hole")
plt.show()
# %%
from glcm_cupy import glcm

radius = 2
step_size = 1
g = glcm(ar, bin_from=256, bin_to=16, radius=radius, step_size=step_size)

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
