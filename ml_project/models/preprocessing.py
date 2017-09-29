from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

class StandardScaler(StandardScaler):
    """standardize data"""
    def __init__(self):
        super(StandardScaler, self).__init__()

    def fit(self, X, y=None):
        super(StandardScaler, self).fit(X[:, 1:100])
        print("mean = ")
        print(self.mean_)
        print("variance = ")
        print(self.var_)
        return self

    def transform(self, X, y='deprecated', copy=None):
        return super(StandardScaler, self).transform(X[:, 1:100])


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

