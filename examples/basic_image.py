""" Example for Basic Image Transformation """
# Here, we load in the array
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.image import imread
from mpl_toolkits.axes_grid1 import ImageGrid

# %%
# Divide the image by 16 to reduce runtime
ar = imread(Path(__file__).parents[1] / "data/image.jpg")[::4, ::4]

# %%
# Option 1: Use the class variant to run GLCM
from glcm_cupy import GLCM, Direction, Features

g = GLCM(
    # Select directions
    directions=(Direction.EAST, Direction.SOUTH_EAST),
    # Select features
    features=(Features.ASM, Features.CONTRAST),
    # Because JPG images span 0 - 255, we bin from 256 (total values)
    # Squash it to 16 values for speed
    bin_from=256, bin_to=16,
    # Whether to scale the values to 0 - 1.
    # This is scaled based on GLCM features' maximum theoretical value.
    # Which means it's independent of scale of input
    # See: glcm_cupy.utils.normalize_features()
    normalized_features=True,
    # Whether to output the progress bar
    verbose=True,
    # Step size between windows.
    # Note that it's a sliding window, thus step_size of 2 doesn't half the
    # size of the resulting array.
    step_size=1,
).run(ar)

# %%
# Alternatively, use the function variant
from glcm_cupy import glcm

g = glcm(ar, bin_from=256, bin_to=16)  # Argument names are the same

# %%
# We yield the features using constants defined in conf
from glcm_cupy.conf import Features

print(g[..., Features.CONTRAST])
print(g[..., Features.CORRELATION])
print(g[..., Features.ASM])
# %%
# Alternatively, since these constants are simply integers
print(g[..., 0])
print(g[..., 1])
print(g[..., 2])
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
fig.suptitle('GLCM Features (Cube Rooted for visibility)')
fig.show()
