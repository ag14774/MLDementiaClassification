import math
import numbers
import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array, check_is_fitted


class NonZeroSelection(BaseEstimator, TransformerMixin):
    """Select non-zero voxels"""

    def fit(self, X, y=None):
        X = check_array(X)
        self.nonzero = X.sum(axis=0) > 0

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["nonzero"])
        X = check_array(X)
        return X[:, self.nonzero]


class RandomSelection(BaseEstimator, TransformerMixin):
    """Random Selection of features"""

    def __init__(self, n_components=1000, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)
        self.components = sample_without_replacement(
            n_features, self.n_components, random_state=random_state)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        n_samples, n_features = X.shape
        X_new = X[:, self.components]

        return X_new


class ShadeExtraction(BaseEstimator, TransformerMixin):
    """Count how many pixels fall in each shade category"""

    def __init__(self,
                 random_state=32,
                 n_shades=10,
                 max_range=None,
                 cube_x=16,
                 cube_y=16,
                 cube_z=16):
        self.random_state = random_state
        self.n_shades = n_shades
        self.cube_x = cube_x
        self.cube_y = cube_y
        self.cube_z = cube_z
        self.max_range = max_range
        self.boundaries = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Shape before bin count: ", X.shape)
        sys.stdout.flush()

        X = check_array(X)
        if self.max_range is None:
            self.max_range = X.max()

        cubetotalsize = self.cube_x * self.cube_y * self.cube_z
        hist_num = math.ceil(X.shape[1] / cubetotalsize)

        if cubetotalsize < self.n_shades:
            raise Exception(
                'cubetotalsize must be greater or equal to bin_num')

        for j in range(0, X.shape[0]):
            for i in range(0, hist_num):
                X[j, i * self.n_shades:(
                    i + 1) * self.n_shades], _ = np.histogram(
                        X[j, i * cubetotalsize:(i + 1) * cubetotalsize],
                        self.n_shades, (1, self.max_range))

        X = X[:, 0:hist_num * self.n_shades]

        print("X after bincount: ")
        print(X)
        sys.stdout.flush()

        return X
