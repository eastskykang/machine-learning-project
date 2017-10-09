import sklearn as skl
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt


class KernelEstimator(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """docstring"""
    def __init__(self, save_path=None):
        super(KernelEstimator, self).__init__()
        self.save_path = save_path

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.y_mean = np.mean(y)
        y -= self.y_mean
        Xt = np.transpose(X)
        cov = np.dot(X, Xt)
        alpha, _, _, _ = np.linalg.lstsq(cov, y)
        self.coef_ = np.dot(Xt, alpha)

        if self.save_path is not None:
            plt.figure()
            plt.hist(self.coef_[np.where(self.coef_ != 0)], bins=50,
                     normed=True)
            plt.savefig(self.save_path + "KernelEstimatorCoef.png")
            plt.close()

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "y_mean"])
        X = check_array(X)

        prediction = np.dot(X, self.coef_) + self.y_mean

        if self.save_path is not None:
            plt.figure()
            plt.plot(prediction, "o")
            plt.savefig(self.save_path + "KernelEstimatorPrediction.png")
            plt.close()

        return prediction

    def score(self, X, y, sample_weight=None):
        scores = (self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)

        if self.save_path is not None:
            plt.figure()
            plt.plot(scores, "o")
            plt.savefig(self.save_path + "KernelEstimatorScore.png")
            plt.close()

            df = pd.DataFrame({"score": scores})
            df.to_csv(self.save_path + "KernelEstimatorScore.csv")

        return score

    def set_save_path(self, save_path):
        self.save_path = save_path

class MLPRegressor(MLPRegressor):
    """Multi-Layer Perceptron"""
    def __init__(self, hidden_layer_sizes=(5000, 1000, 500, 100), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate='constant', learning_rate_init=0.001, shuffle=True, random_state=None):
        super(MLPRegressor, self).__init__(self, hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                                           alpha=0.0001, batch_size=batch_size, learning_rate=learning_rate,
                                           learning_rate_init=learning_rate_init, shuffle=shuffle, random_state=random_state)

    def fit(self, X, y=None):
        super(MLPRegressor, self).fit(X, y)

    def transform(self, X, y=None):
        super(MLPRegressor, self).transform(X, y)

