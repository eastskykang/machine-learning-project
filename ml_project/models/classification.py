import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

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


class LogisticRegressionWithLabelAssignment(LogisticRegression):
    """Logistic Regression"""
    def __init__(self, solver='lbfgs', multi_class='multinomial'):
        super(LogisticRegressionWithLabelAssignment, self).__init__(
            solver=solver,
            multi_class=multi_class)

    def fit(self, X, y, sample_weight=None):
        # assign label by argmax
        y_assigned = np.argmax(y, axis=1)
        X, y_assigned = check_X_y(X, y_assigned)

        super(LogisticRegressionWithLabelAssignment, self)\
            .fit(X, y_assigned, sample_weight)
        return self

    def score(self, X, y, sample_weight=None):
        P_predicted = self.predict_proba(X)
        n_samples, n_labels = np.shape(P_predicted)

        score = np.zeros(n_samples)

        for i in range(0, n_samples):
            score[i] = spearmanr(y[i, :], P_predicted[i, :])

        return np.mean(score)

    def predict_proba(self, X):
        return super(LogisticRegressionWithLabelAssignment, self)\
            .predict_proba(X)