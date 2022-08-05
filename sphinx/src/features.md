# GLCM Feature Indexing

(select_feature)=

## Selecting Features

To reduce run time, you can select specific features.

```{note}
Unselected features returned will be zeroed (or 0.5 if normalized correlation).
```

```{note}
Note that the order of ``features`` does not determine the order of ``g``'s channel.
```

```pycon
>>> from glcm_cupy import GLCM, CONTRAST, CORRELATION
>>> import cv2
>>> ar = cv2.imread("image.jpg")
>>> g = GLCM(..., features=(CONTRAST, CORRELATION)).run(ar)
```

(get_feature)=

## Getting Features

To retrieve the features, use the following syntax

```pycon
>>> from glcm_cupy import Features
>>> print(g[..., Features.CONTRAST])
>>> print(g[..., Features.CORRELATION])
```

## Feature Theory

```{note}
See Definitions from [GLCM Texture: A Tutorial v. 3.0 March 2017](https://prism.ucalgary.ca/handle/1880/51900) 
```

In total, we have 6 features.

```{math}

\text{Homogeneity} = \sum_{i,j=0}^{N-1}\frac{P_{i,j}}{1+(i-j)^2}\\
\text{Contrast} = \sum_{i,j=0}^{N-1}P_{i,j}(i-j)^2\\
\text{Angular Second Moment (ASM)} = \sum_{i,j=0}^{N-1}P_{i,j}^2\\
\text{GLCM Mean, } \mu = \sum_{i,j=0}^{N-1}i * P_{i,j}\\
\text{GLCM Variance, } \sigma^2 = \sum_{i,j=0}^{N-1}P_{i,j}(i - \mu_i)^2\\
\text{Correlation} = \sum_{i,j=0}^{N-1}P_{i,j}\frac{(i - \mu_i)(j - \mu_j)}{\sqrt{\sigma_i^2\sigma_j^2}}\\
\text{Dissimilarity} = \sum_{i,j=0}^{N-1}P_{i,j} * \left\lvert i - j \right\rvert\\
```

Feel free to suggest more features in the GitHub Issue page.
