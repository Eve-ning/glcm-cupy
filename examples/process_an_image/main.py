import numpy as np
from tqdm.autonotebook import get_ipython

from glcm_cupy import GLCM
from PIL import Image

ar = np.asarray(Image.open("image.jpg"))[::3,::3]
g = GLCM(bin_from=256, bin_to=16).from_3dimage(ar)
