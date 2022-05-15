import numpy as np
from skimage.feature import graycoprops, graycomatrix

def glcm_py_skimage(i, j):
    ar_ij = np.stack([i, j], axis=-1)
    ar_ji = np.stack([j, i], axis=-1)
    glcm_ij = graycomatrix(ar_ij, (1,), (0,))
    glcm_ji = graycomatrix(ar_ji, (1,), (0,))
    glcm = (glcm_ij + glcm_ji) / 2

    contrast = float(graycoprops(glcm, 'contrast').squeeze())
    homogeneity = float(graycoprops(glcm, 'homogeneity').squeeze())
    asm = float(graycoprops(glcm, 'ASM').squeeze())
    correlation = float(graycoprops(glcm, 'correlation').squeeze())

    return dict(
        homogeneity=homogeneity,
        contrast=contrast,
        asm=asm,
        correlation=correlation
    )
