import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
from sklearn.feature_selection.univariate_selection import _clean_nans
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import mapProbabilitiesToClasses


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


class SelectPercentile2(SelectPercentile):
    """docstring"""

    def __init__(self,
                 score_func=f_classif,
                 percentile=10,
                 yMapMethod='argmax'):
        super(SelectPercentile2, self).__init__(score_func, percentile)
        self.yMapMethod = str(yMapMethod)

    def fit(self, X, y):
        ycopy = y.copy()
        if self.yMapMethod not in ['probability_map', 'argmax']:
            raise Exception("Unsupported y mapping method")
        if self.yMapMethod == 'probability_map':
            ycopy = mapProbabilitiesToClasses(ycopy, 10)
        elif self.yMapMethod == 'argmax':
            ycopy = np.argmax(ycopy, axis=1)
        super(SelectPercentile2, self).fit(X, ycopy)
        return self


class SelectKBestGroups(SelectKBest):
    """docstring"""

    def __init__(self,
                 score_func=f_classif,
                 k=10,
                 group_size=10,
                 yMapMethod='argmax'):
        super(SelectKBestGroups, self).__init__(score_func, k)
        self.yMapMethod = str(yMapMethod)
        self.group_size = group_size

    def fit(self, X, y):
        ycopy = y.copy()
        if self.yMapMethod not in ['probability_map', 'argmax']:
            raise Exception("Unsupported y mapping method")
        if self.yMapMethod == 'probability_map':
            ycopy = mapProbabilitiesToClasses(ycopy, 10)
        elif self.yMapMethod == 'argmax':
            ycopy = np.argmax(ycopy, axis=1)
        super(SelectKBestGroups, self).fit(X, ycopy)
        return self

    def transform(self, X):
        res = super(SelectKBestGroups, self).transform(X)
        print("Shape after SelectKBestGroups: ", res.shape)
        return res

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')

        if self.k == 'all':
            return np.ones(self.scores_.shape, dtype=bool)
        elif self.k == 0:
            return np.zeros(self.scores_.shape, dtype=bool)
        else:
            scores = _clean_nans(self.scores_)
            numofgroups = math.ceil(len(scores) / self.group_size)
            for i in range(0, numofgroups):
                meanScoreOfGroup = np.mean(
                    scores[i * self.group_size:(i + 1) * self.group_size])
                # Whole cube is assigned the mean its elements
                scores[i * self.group_size:(
                    i + 1) * self.group_size] = meanScoreOfGroup

            mask = np.zeros(scores.shape, dtype=bool)

            # Request a stable sort. Mergesort takes more memory (~40MB per
            # megafeature on x86-64).
            mask[np.argsort(scores,
                            kind="mergesort")[-self.k * self.group_size:]] = 1
            return mask
