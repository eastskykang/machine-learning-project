import numpy as np
import sklearn as skl
import pickle

class RandomSelection(skl.base.BaseEstimator, skl.base.TransformerMixin):

    def __init__(self, n_components=1000, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

        self.components = None

    def fit(self, data):
        pass

    def transform(self, data):
        pass

    def fit_transform(self, data):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass


def load_data(path):
    return pickle.load(open(path, "rb"))


def save_data(data, path):
     return pickle.dump(data, open(path, "wb"))


class Data(object):
    """docstring for Data"""
    def __init__(self, X=None, y=None, pipeline=None):
        self.X = X
        self.y = y
        self.pipeline = pipeline
        