from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
from sklearn.feature_selection import VarianceThreshold


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
                            n_features,
                            self.n_components,
                            random_state=random_state)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        n_samples, n_features = X.shape
        X_new = X[:, self.components]

        return X_new


class VarianceThreshold(VarianceThreshold):
    """VarianceThreshold"""
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        super(VarianceThreshold, self).__init__(self.threshold)

    def fit(self, X, y=None):
        print("------------------------------------")
        print("VarianceThreshold fit with thr = {}"
              .format(self.threshold))

        X = check_array(X)
        super(VarianceThreshold, self).fit(X)
        return self

    def transform(self, X, y=None):
        print("------------------------------------")
        print("VarianceThreshold transform with thr = {}"
              .format(self.threshold))

        X = check_array(X)
        print("shape before variance threshold: ")
        print(X.shape)

        X_new = super(VarianceThreshold, self).transform(X)
        print("shape after variance threshold: ")
        print(X_new.shape)

        return X_new
