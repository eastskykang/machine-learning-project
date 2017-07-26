import numpy as np
import sklearn as skl

class RandomSelection(skl.base.BaseEstimator, skl.base.TransformerMixin):

    def __init__(self, n_components=1000):
        self.n_components = n_components

    def fit(self, X):
        pass

    def fit_transform(self, X):
        pass

    def get_params(self):
        pass

    def set_params(self):
        passgit push 