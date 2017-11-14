import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import (mapClassToProbabilities,
                                     mapProbabilitiesToClasses, post_process_y,
                                     post_process_y2, scorer)


class LogisticAdaBoost(AdaBoostClassifier):
    """doctstring"""

    def __init__(self,
                 n_estimators=10,
                 learning_rate=1.0,
                 algorithm='SAMME.R',
                 verbose=0,
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
                 warm_start_logistic=False,
                 post_process_method='simple',
                 post_process_threshold=0.35,
                 post_process_min_samples=2,
                 post_process_eps=0.03):
        base_estimator = LogisticRegression(
            penalty, dual, tol, C, fit_intercept, intercept_scaling,
            class_weight, random_state, solver, max_iter, multi_class, verbose,
            warm_start_logistic, 1)
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
        self.warm_start_logistic = warm_start_logistic
        self.n_jobs = 1
        super(LogisticAdaBoost,
              self).__init__(base_estimator, n_estimators, learning_rate,
                             algorithm, random_state)
        self.post_process_method = post_process_method
        self.post_process_threshold = post_process_threshold
        self.post_process_min_samples = post_process_min_samples
        self.post_process_eps = post_process_eps

    def fit(self, X, y):
        self.base_estimator.set_params(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            verbose=self.verbose,
            warm_start_logistic=self.warm_start_logistic,
            n_jobs=self.n_jobs)
        print(self)
        n_samples, n_features = X.shape
        n_labels = y.shape[1]
        X = np.repeat(X, n_labels, 0)
        weights = y.reshape(-1)
        y = np.tile(np.arange(n_labels), n_samples)
        super(LogisticAdaBoost, self).fit(X, y, weights)
        return self

    def predict_proba(self, X):
        ypred = super(LogisticAdaBoost, self).predict_proba(X)
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


class LogisticBagging(BaggingClassifier):
    """doctstring"""

    def __init__(self,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start_bagging=False,
                 n_jobs=1,
                 verbose=0,
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
                 warm_start_logistic=False,
                 post_process_method='simple',
                 post_process_threshold=0.35,
                 post_process_min_samples=2,
                 post_process_eps=0.03):
        base_estimator = LogisticRegression(
            penalty, dual, tol, C, fit_intercept, intercept_scaling,
            class_weight, random_state, solver, max_iter, multi_class, verbose,
            warm_start_logistic, 1)
        super(LogisticBagging, self).__init__(
            base_estimator, n_estimators, max_samples, max_features, bootstrap,
            bootstrap_features, oob_score, warm_start_bagging, n_jobs,
            random_state, verbose)
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
        super(LogisticBagging, self).fit(X, y, weights)
        return self

    def predict_proba(self, X):
        ypred = super(LogisticBagging, self).predict_proba(X)
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


class RandomForestClassifier2(RandomForestClassifier):
    def __init__(self,
                 n_estimators=10,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_split=1e-07,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 post_process_threshold=0.3):
        super(RandomForestClassifier2, self).__init__(
            n_estimators, criterion, max_depth, min_samples_split,
            min_samples_leaf, min_weight_fraction_leaf, max_features,
            max_leaf_nodes, min_impurity_split, bootstrap, oob_score, n_jobs,
            random_state, verbose, warm_start, class_weight)
        self.post_process_threshold = post_process_threshold

    def fit(self, X, y):
        print(self)
        n_samples, n_features = X.shape
        n_labels = y.shape[1]
        X = np.repeat(X, n_labels, 0)
        weights = y.reshape(-1)
        y = np.tile(np.arange(n_labels), n_samples)
        super(RandomForestClassifier2, self).fit(X, y, weights)
        return self

    def predict_proba(self, X):
        ypred = super(RandomForestClassifier2, self).predict_proba(X)
        return post_process_y2(ypred, threshold=self.post_process_threshold)

    def score(self, X, y):
        return scorer(self, X, y)


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
            penalty, dual, tol, C, fit_intercept, intercept_scaling,
            class_weight, random_state, solver, max_iter, multi_class, verbose,
            warm_start, n_jobs)
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
