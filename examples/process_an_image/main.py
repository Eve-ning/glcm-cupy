""" Example Code """
#%%
import numpy as np
from PIL import Image

# Here, we load in the array
ar = np.asarray(Image.open("image.jpg"))[::4,::4]

#%%
# We may use the class variant to run GLCM
from glcm_cupy import GLCM

g = GLCM(bin_from=256, bin_to=16).run(ar)

#%%
# Alternatively, use the function variant
from glcm_cupy import glcm

g = glcm(ar, bin_from=256, bin_to=16)

#%%
# We yield the features using constants defined in conf
from glcm_cupy.conf import CONTRAST, CORRELATION, ASM

g[..., CONTRAST]
g[..., CORRELATION]
g[..., ASM]
#%%
# Alternatively, since these constants are simply integers
g[..., 0]
g[..., 1]
g[..., 2]

