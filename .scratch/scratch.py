import numpy as np
from PIL import Image
from tifffile import tifffile


from definitions import ROOT_DIR
from glcm_cuda import GLCM
# #%%
from tests.unit_tests import glcm_expected

img = Image.open(f"{ROOT_DIR}/data/sample.jpg")
ar = np.asarray(img)[::10, :: 10]
#%%
g = GLCM(bins=16).from_3dimage(ar)
