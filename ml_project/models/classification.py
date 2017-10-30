import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class AssignLabelsClassifier(BaseEstimator, TransformerMixin):
    """docstring"""
    def __init__(self, base_estimator, **kwargs):
        self.base_estimator = base_estimator(**kwargs)

    def fit(self, X, y):
        ylabels = np.argmax(y, axis=1)
        self.base_estimator.fit(X, ylabels)
        return self

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def score(self, X, y):
        ypred = self.predict_proba(X)
        corrs = np.zeros(X.shape[0])
        for i in range(0, X.shape[0]):
            corrs[i] = spearmanr(y[i], ypred[i]).correlation

        meanrho = np.mean(corrs)

        print(meanrho)
        sys.stdout.flush()

        return meanrho


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
