from itertools import chain, combinations

import numpy as np
import pytest

from glcm_cupy import GLCM, Features
from glcm_cupy.glcm_base import FEATURES


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(
        chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    )


@pytest.mark.parametrize(
    "features", powerset(FEATURES), ids=[str(i) for i in powerset(FEATURES)]
)
def test_glcm_feature_select(features):
    features = set(features)
    ar = np.random.randint(0, 256, [15, 15, 1])
    if not features:
        with pytest.raises(ValueError):
            GLCM(radius=3, bin_from=256, bin_to=16, features=features).run(ar)
    else:
        g = GLCM(radius=3, bin_from=256,
                 bin_to=16, features=features,
                 normalized_features=False).run(ar)
        if Features.CORRELATION in features:
            features.add(Features.VARIANCE)
        if Features.VARIANCE in features:
            features.add(Features.MEAN)
        features = tuple(features)

        assert (g[..., features] != 0).flatten().all()
        assert (g[..., [i for i in FEATURES
                        if i not in features]] == 0).flatten().all()
