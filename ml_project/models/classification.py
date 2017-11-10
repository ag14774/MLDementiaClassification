import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import (mapClassToProbabilities,
                                     mapProbabilitiesToClasses, post_process_y,
                                     post_process_y2, scorer)


class LogisticRegression2(LogisticRegression):
    """doctstring"""

    def __init__(self,
                 penalty='l2',
                 dual=False,
                 tol=0.0001,
                 C=1.0,
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 random_state=None,
                 solver='liblinear',
                 max_iter=100,
                 multi_class='ovr',
                 verbose=0,
                 warm_start=False,
                 n_jobs=1,
                 post_process_method='simple',
                 post_process_threshold=0.35,
                 post_process_min_samples=2,
                 post_process_eps=0.03):
        super(LogisticRegression2, self).__init__(
            penalty='l2',
            dual=False,
            tol=0.0001,
            C=1.0,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=None,
            solver='liblinear',
            max_iter=100,
            multi_class='ovr',
            verbose=0,
            warm_start=False,
            n_jobs=1)
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.post_process_method = post_process_method
        self.post_process_threshold = post_process_threshold
        self.post_process_min_samples = post_process_min_samples
        self.post_process_eps = post_process_eps

    def fit(self, X, y):
        print(self)
        n_samples, n_features = X.shape
        n_labels = y.shape[1]
        X = np.repeat(X, n_labels, 0)
        weights = y.reshape(-1)
        y = np.tile(np.arange(n_labels), n_samples)
        super(LogisticRegression2, self).fit(X, y, weights)
        return self

    def predict_proba(self, X):
        ypred = super(LogisticRegression2, self).predict_proba(X)
        if self.post_process_method == "simple":
            return post_process_y2(
                ypred, threshold=self.post_process_threshold)
        elif self.post_process_method == "advanced":
            return post_process_y(
                ypred,
                eps=self.post_process_eps,
                min_samples=self.post_process_min_samples)
        else:
            return ypred

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
        self.dictargs.update(self.kwargs)
        self.base_estimator = self.base_estimator.set_params(**self.dictargs)
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
