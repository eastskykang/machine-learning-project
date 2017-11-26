from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array


class StandardScaler(StandardScaler):
    """standardize data"""
    def __init__(self, verbosity=0):
        super(StandardScaler, self).__init__()
        self.verbosity = verbosity

    def fit(self, X, y=None):
        if self.verbosity > 0:
            print("------------------------------------")
            print("StandardScaler fit")

        super(StandardScaler, self).fit(X)

        if self.verbosity > 0:
            print("before standardized: ")
            print("mean = ")
            print(self.mean_)
            print("variance = ")
            print(self.var_)

        return self

    def transform(self, X, y='deprecated', copy=None):
        if self.verbosity > 0:
            print("------------------------------------")
            print("StandardScaler transform")

        X_new = super(StandardScaler, self).transform(X)

        if self.verbosity > 0:
            print("after standardized: ")
            print("mean = ")
            print(X_new.mean(axis=0))
            print("variance = ")
            print(X_new.std(axis=0))

        return X_new


class PrincipleComponentAnalysis(PCA):

    def __init__(self, n_components=None):
        super(PrincipleComponentAnalysis, self).__init__(
            n_components=n_components)

    def fit(self, X, y=None):

        X = check_array(X)

        print("------------------------------------")
        print("PCA fit")
        print("n_components = {}".format(self.n_components))
        print("variances = {}".format(self.explained_variance_))

        super(PrincipleComponentAnalysis, self).fit(X, y)
        return self

    def transform(self, X, y=None):

        X = check_array(X)

        print("------------------------------------")
        print("PCA transform")
        print("shape of X before pca : ")
        print(X.shape)

        X_new = super(PrincipleComponentAnalysis, self).transform(X, y)

        print("shape of X after pca : ")
        print(X_new.shape)

        return X_new

