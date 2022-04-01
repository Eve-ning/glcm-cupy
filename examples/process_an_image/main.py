""" Example Code """
#%%
import numpy as np
from PIL import Image


# Here, we load in the array
# We divide the image by / 16 as it'll take too long

ar = np.asarray(Image.open("../../data/image.jpg"))[::4,::4]

#%%
# We may use the class variant to run GLCM
from glcm_cupy import GLCM, Direction

g = GLCM(
    directions=(Direction.EAST, Direction.SOUTH_EAST),
    bin_from=256, bin_to=16).run(ar)

#%%
# Alternatively, use the function variant
from glcm_cupy import glcm

g = glcm(ar, bin_from=256, bin_to=16)

#%%
# We yield the features using constants defined in conf
from glcm_cupy.conf import CONTRAST, CORRELATION, ASM

print(g[..., CONTRAST])
print(g[..., CORRELATION])
print(g[..., ASM])
#%%
# Alternatively, since these constants are simply integers
print(g[..., 0])
print(g[..., 1])
print(g[..., 2])
