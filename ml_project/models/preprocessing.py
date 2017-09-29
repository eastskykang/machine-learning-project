from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

class Standardization(BaseEstimator, TransformerMixin):
    """standardize data"""
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        print("mean")
        print(self.scaler.mean_)
        print("variance")
        print(self.scaler.var_)
        return self

    def transform(self, X, y=None):
        return self.scaler.transform(X)


class PrincipleComponentAnalysis(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=100):
        self.n_components = n_components
        self.pca = PCA(n_components)

    def fit(self, X, y=None):
        self.pca.fit(X);

        print("\nPCA with n_components = {}".format(self.n_components))
        print("variances = {}".format(self.pca.explained_variance_))

        return self

    def transform(self, X, y=None):
        return self.pca.transform(self, X)

