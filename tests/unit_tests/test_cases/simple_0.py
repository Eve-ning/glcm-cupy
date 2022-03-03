import cupy as cp

"""
homogeneity p / (1 + (i - j)^2)
contrast    p * (i - j)^2
asm         p^2
mean_i      p * i
mean_j      p * j
var_i       p * (i - MEAN_I)^2 
var_j       p * (j - MEAN_J)^2
correlation p * (i - MEAN_I) * (j - MEAN_J) / sqrt(VAR_I * VAR_J)
"""

simple_0 = [
    # GLCM
    #     j j
    #     0 1
    # i 0 0 1
    # i 1 0 0
    cp.asarray([0, 0, 0, 0]),
    cp.asarray([1, 1, 1, 1]),
    0.5,  # homogeneity
    1,  # contrast
    1,  # asm
    0,  # mean_i
    1,  # mean_j
    0,  # var_i
    0,  # var_j
    0,  # correlation
]
