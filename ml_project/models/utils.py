import sys

from sklearn.base import BaseEstimator, TransformerMixin


class Debugger(BaseEstimator, TransformerMixin):
    """Debugger"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Shape of X: " + str(X.shape))
        if y is not None:
            print("Shape of y: " + str(y.shape))
        sys.stdout.flush()
        return X
