import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectPercentile, f_classif
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
            ycopy = mapProbabilitiesToClasses(ycopy, 100)
        elif self.yMapMethod == 'argmax':
            ycopy = np.argmax(ycopy, axis=1)
        super(SelectPercentile2, self).fit(X, ycopy)
        return self
