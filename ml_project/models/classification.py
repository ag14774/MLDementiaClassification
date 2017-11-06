import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import (
    StratifiedKFoldProbLabels, mapClassToProbabilities,
    mapProbabilitiesToClasses, packY, scorer, unpackY)


class MetaClassifierRFE(RFECV):
    """docstring"""

    def __init__(self, step, n_splits, shuffle, random_state, base_estimator,
                 **base_estimator_args):
        self.estimator = MetaClassifier(base_estimator, **base_estimator_args)
        self.step = step
        self.cv = StratifiedKFoldProbLabels(n_splits, shuffle, random_state)
        self.scoring = scorer
        self.verbose = 0
        self.n_jobs = 1

    def fit(self, X, y):
        y = packY(y)
        super(MetaClassifierRFE, self).fit(X, y)
        print("Selected " + str(self.n_features_))
        return self


class MetaClassifier(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, base_estimator, dictargs={}, **kwargs):
        if str(type(base_estimator)) == "<class 'type'>":
            self.base_estimator = base_estimator()
        else:
            self.base_estimator = base_estimator
        self.dictargs = dictargs
        self.kwargs = kwargs

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
        self.dictargs.update(self.kwargs)
        self.base_estimator = self.base_estimator.set_params(**self.dictargs)
        print(self.base_estimator)
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


class MetaClassifierProbabilityMap(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, base_estimator, dictargs=None, **kwargs):
        if str(type(base_estimator)) == "<class 'type'>":
            self.base_estimator = base_estimator()
        else:
            self.base_estimator = base_estimator
        self.dictargs = dictargs
        self.kwargs = kwargs

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
        if self.dictargs is None:
            self.base_estimator = self.base_estimator.set_params(**self.kwargs)
        else:
            print(self.dictargs)
            self.base_estimator = self.base_estimator.set_params(
                **self.dictargs)
        print(self.base_estimator)
        y = mapProbabilitiesToClasses(y, 10)
        self.base_estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        pred = self.base_estimator.predict_proba(X)
        # print("Classes: ", self.base_estimator.classes_.astype(int))
        probs = mapClassToProbabilities(
            self.base_estimator.classes_.astype(int), minlength=4, base=10)
        # print("Probability map: ", probs)
        newProb = np.zeros((pred.shape[0], 4))
        for i in range(0, pred.shape[0]):
            for j in range(0, pred.shape[1]):
                temp = np.reshape(pred[i, j] * probs[j], -1)
                newProb[i, :] += temp
            # print("Before normalising: ", newProb[i])
            newProb[i] = newProb[i] / sum(newProb[i])
            # print("After normalising: ", newProb[i])
        inds = np.where(np.isnan(newProb))
        newProb[inds] = 0.25
        newProb = np.nan_to_num(newProb, False)
        newProb[newProb == 0] = 1e-200
        return newProb

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
