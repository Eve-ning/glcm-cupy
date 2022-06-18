import numpy as np

try:
    # Try to import scikit-image 0.19+
    from skimage.feature import greycoprops
    from skimage.feature import greycomatrix
except:
    from skimage.feature import graycoprops as greycoprops
    from skimage.feature import graycomatrix as greycomatrix


def glcm_py_skimage(i, j):
    ar_ij = np.stack([i, j], axis=-1)
    ar_ji = np.stack([j, i], axis=-1)
    glcm_ij = greycomatrix(ar_ij, (1,), (0,))
    glcm_ji = greycomatrix(ar_ji, (1,), (0,))
    glcm = (glcm_ij + glcm_ji) / 2

    contrast = float(greycoprops(glcm, 'contrast').squeeze())
    homogeneity = float(greycoprops(glcm, 'homogeneity').squeeze())
    asm = float(greycoprops(glcm, 'ASM').squeeze())
    correlation = float(greycoprops(glcm, 'correlation').squeeze())

    return dict(
        homogeneity=homogeneity,
        contrast=contrast,
        asm=asm,
        correlation=correlation
    )
