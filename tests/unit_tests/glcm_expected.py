import numpy as np


def glcm_expected(i: np.ndarray,
                  j: np.ndarray):
    """ Calculate the expected GLCM features, albeit slower.

    Notes:
        This is to support the tests in asserting the actual values.
        This can be technically used in production however, it's very slow.

    Args:
        i: Window I, may be ndim
        j: Window J, may be ndim

    Returns:
        Dictionary of these keys
            (homogeneity, contrast, asm, mean_i, mean_j,
             var_i, var_j, correlation).
        Values as float.
    """
    i_flat = i.flatten()
    j_flat = j.flatten()
    assert len(i_flat) == len(j_flat)
    glcm_size = max(max(i_flat) + 1, max(j_flat) + 1)
    glcm = np.zeros((glcm_size, glcm_size), dtype=float)

    for i_, j_ in zip(i_flat, j_flat):
        glcm[i_, j_] += 1

    glcm /= len(i_flat)

    homogeneity = 0.0
    contrast = 0.0
    asm = 0.0
    mean_i = 0.0
    mean_j = 0.0
    var_i = 0.0
    var_j = 0.0
    correlation = 0.0

    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
            contrast += glcm[i, j] * (i - j) ** 2
            asm += glcm[i, j] ** 2
            mean_i += glcm[i, j] * i
            mean_j += glcm[i, j] * j

    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            var_i += glcm[i, j] * (i - mean_i) ** 2
            var_j += glcm[i, j] * (j - mean_j) ** 2

    if var_i != 0 and var_j != 0:
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                correlation += \
                    glcm[i, j] \
                    * (i - mean_i) \
                    * (j - mean_j) \
                    / ((var_i * var_j) ** 0.5)

    return dict(
        homogeneity=homogeneity,
        contrast=contrast,
        asm=asm,
        mean_i=mean_i,
        mean_j=mean_j,
        var_i=var_i,
        var_j=var_j,
        correlation=correlation
    )
