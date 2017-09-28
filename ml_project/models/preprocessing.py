# from sklearn import preprocessing
#
# class Standardization():
#     """standardize data"""
#     def __init__(self, n_components=1000, random_state=None):
#         self.n_components = n_components
#         self.random_state = random_state
#         self.components = None
#
#     def fit(self, X, y=None):
#         X = check_array(X)
#         n_samples, n_features = X.shape
#
#         random_state = check_random_state(self.random_state)
#         self.components = sample_without_replacement(
#                             n_features,
#                             self.n_components,
#                             random_state=random_state)
#         return self
#
#     def transform(self, X, y=None):
#         check_is_fitted(self, ["components"])
#         X = check_array(X)
#         n_samples, n_features = X.shape
#         X_new = X[:, self.components]
#
#         return X_new
#
#
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

class PrincipleComponentAnalysis(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=100):
        self.n_components = n_components
        self.pca = PCA(n_components)

    def fit(self, X, y=None):
        self.pca.fit(self, X, y);
        return self

    def transform(self, X, y=None):
        return self.pca.transform(self, X, y)

