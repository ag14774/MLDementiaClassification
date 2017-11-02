import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold


def scorer(estimator, X, y):
    y = unpackY(y)
    ypred = estimator.predict_proba(X)
    corrs = np.zeros(X.shape[0])
    for i in range(0, X.shape[0]):
        corrs[i] = spearmanr(y[i], ypred[i]).correlation
    meanrho = np.mean(corrs)
    print(meanrho)
    sys.stdout.flush()
    return meanrho


def packY(y):
    ynew = np.zeros(y.shape[0], dtype=(tuple))
    for i, row in enumerate(y):
        ynew[i] = tuple(row)
    return ynew


def unpackY(y):
    ynew = np.zeros((y.shape[0], len(y[0])))
    for i, row in enumerate(y):
        ynew[i] = np.array(row)
    return ynew


class Debugger(BaseEstimator, TransformerMixin):
    """Debugger"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Shape of X: " + str(X.shape))
        sys.stdout.flush()
        return X


class StratifiedKFoldProbLabels(StratifiedKFold):
    """StratifiedKFold that handles probabilities as labels"""

    def split(self, X, y, groups=None):
        ycopy = y.copy()
        ycopy = unpackY(ycopy)
        ycopy = np.argmax(ycopy, axis=1)
        return super(StratifiedKFoldProbLabels, self).split(X, ycopy, groups)
