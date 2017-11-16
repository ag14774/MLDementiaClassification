import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import (RFE, SelectKBest, SelectPercentile,
                                       f_classif)
from sklearn.feature_selection.univariate_selection import _clean_nans
from sklearn.utils import check_random_state, check_X_y, safe_sqr
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array, check_is_fitted

from ml_project.models.utils import mapProbabilitiesToClasses


class RFEWithSampleWeights(RFE):
    def __init__(self,
                 estimator,
                 n_features_to_select=None,
                 step=1,
                 verbose=0):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit the RFE model and then the underlying estimator on the selected
           features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """
        return self._fit(X, y, sample_weight=sample_weight)

    def _fit(self, X, y, step_score=None, sample_weight=None):
        # Parameter step_score controls the calculation of self.scores_
        # step_score is not exposed to users
        # and is used when implementing RFECV
        # self.scores_ will not be calculated when calling _fit through fit

        X, y = check_X_y(X, y, "csc")
        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        # Elimination
        while np.sum(support_) > n_features_to_select:
            print("Features remaining: ", np.sum(support_))
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            estimator = clone(self.estimator)
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            estimator.fit(X[:, features], y, sample_weight)

            # Get coefs
            if hasattr(estimator, 'coef_'):
                coefs = estimator.coef_
            else:
                coefs = getattr(estimator, 'feature_importances_', None)
            if coefs is None:
                raise RuntimeError('The classifier does not expose '
                                   '"coef_" or "feature_importances_" '
                                   'attributes')

            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step, np.sum(support_) - n_features_to_select)

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y, sample_weight)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self


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

    def __init__(self, score_func=f_classif, percentile=10):
        super(SelectPercentile2, self).__init__(score_func, percentile)

    def fit(self, X, y):
        ycopy = y.copy()
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
