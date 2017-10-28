import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
import tensorflow as tf

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


class LogisticRegression(LogisticRegression):
    """Logistic Regression"""
    def __init__(self, solver='lbfgs', multi_class='multinomial', C=1):
        super(LogisticRegression, self).__init__(
            penalty='l2',
            solver=solver,
            C=C,
            multi_class=multi_class,
            n_jobs=-1)

    def fit(self, X, y, sample_weight=None):
        # assign label by argmax
        y_assigned = np.argmax(y, axis=1)
        X, y_assigned = check_X_y(X, y_assigned)

        super(LogisticRegression, self)\
            .fit(X, y_assigned, sample_weight)
        return self

    def score(self, X, y, sample_weight=None):
        P_predicted = self.predict_proba(X)
        n_samples, n_labels = np.shape(P_predicted)

        score = np.zeros(n_samples)

        for i in range(0, n_samples):
            score[i] = spearmanr(y[i, :], P_predicted[i, :])[0]

        return np.mean(score)

    def predict_proba(self, X):
        return super(LogisticRegression, self)\
            .predict_proba(X)


class LogisticRegressionWithProbability(BaseEstimator, TransformerMixin):

    def __init__(self, learning_rate=0.01, epoch=10000, C=1, verbosity=1):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.verbosity = verbosity
        self.C = C

    def fit(self, X, y, sample_weight=None):
        weight, bias = self.model(X_train=X, y_train=y)
        self.parameters_ = [weight, bias]
        return self

    def model(self, X_train, y_train):
        # dimension
        n_samples, n_features = np.shape(X_train)
        _, n_classes = np.shape(y_train)

        # data
        X = tf.placeholder(tf.float32, shape=[None, n_features])
        Y = tf.placeholder(tf.float32, shape=[None, n_classes])

        # logit hypothesis
        W = tf.Variable(
            tf.random_normal([n_features, n_classes]), name='weight')
        b = tf.Variable(
            tf.random_normal([n_classes]), name='bias')

        logits = tf.matmul(X, W) + b

        # cost function: softmax cross entropy
        C = tf.placeholder(tf.float32, shape=())

        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y,
                                                    logits=logits))

        # add regularizer
        regularizer = tf.nn.l2_loss(W)
        cost = C * cost + regularizer

        train = tf.train.\
            GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # train
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.epoch):
                cost_val, _ = sess.run([cost, train],
                                       feed_dict={X: X_train,
                                                  Y: y_train,
                                                  C: self.C})

                if self.verbosity > 0 and (epoch % 100) == 0:
                    print(epoch, cost_val)

            # parameter return
            weight, bias = sess.run([W, b])

        return weight, bias

    def score(self, X, y, sample_weight=None):
        P_predicted = self.predict_proba(X)
        n_samples, n_labels = np.shape(P_predicted)

        score = np.zeros(n_samples)

        for i in range(0, n_samples):
            score[i] = spearmanr(y[i, :], P_predicted[i, :])[0]

        return np.mean(score)

    def predict_proba(self, X):
        # parameters
        [weight, bias] = self.parameters_

        # logits
        logits_val = np.add(np.matmul(X, weight), bias)
        n_samples, n_classes = np.shape(logits_val)

        # get probability by softmax
        logits = tf.placeholder(tf.float32, shape=[None, n_classes])
        P = tf.nn.softmax(logits)

        with tf.Session() as sess:
            P_predicted = sess.run(P, feed_dict={logits: logits_val})

        return P_predicted