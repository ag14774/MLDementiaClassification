import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.feature_selection import RFECV
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import scorer, packY, unpackY


class MetaClassifierRFE(RFECV):
    """docstring"""

    def __init__(self, cv, n_splits, shuffle, random_state, base_estimator,
                 **base_estimator_args):
        self.estimator = MetaClassifier(base_estimator, **base_estimator_args)
        self.step = 1
        self.cv = cv(n_splits, shuffle, random_state)
        self.scoring = scorer
        self.verbose = 0
        self.n_jobs = 1

    def fit(self, X, y):
        y = packY(y)
        super(MetaClassifierRFE, self).fit(X, y)
        return self


class MetaClassifier(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, base_estimator, **kwargs):
        if str(type(base_estimator)) == "<class 'type'>":
            self.base_estimator = base_estimator()
        else:
            self.base_estimator = base_estimator
        self.base_estimator = self.base_estimator.set_params(**kwargs)

    @property
    def coef_(self):
        return self.base_estimator.coef_

    @property
    def intercept_(self):
        return self.base_estimator.intercept_

    @property
    def n_iter_(self):
        return self.base_estimator.n_iter_

    def fit(self, X, y):
        y = unpackY(y)
        n_samples, n_features = X.shape
        n_labels = y.shape[1]
        X = np.repeat(X, n_labels, 0)
        weights = y.reshape(-1)
        y = np.tile(np.arange(n_labels), n_samples)
        self.base_estimator.fit(X, y, weights)
        return self

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def score(self, X, y):
        return scorer(self, X, y)


class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""

    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))
