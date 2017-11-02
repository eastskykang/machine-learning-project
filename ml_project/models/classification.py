import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout, \
    l1_regularizer, l2_regularizer, flatten
from datetime import datetime
from pathlib import Path
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

        super(LogisticRegression, self) \
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
        return super(LogisticRegression, self) \
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

        train = tf.train. \
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


class NeuralNetClassifier(BaseEstimator, TransformerMixin):

    def __init__(self, save_path=None, hidden_layers=None, activations=None, regularizer='l2',
                 regularizer_scale=1.0, batch_normalization=True,
                 batch_size=58, dropout=True, dropout_rate=0.3,
                 optimizer='Adam', learning_rate=0.01, num_epoch=500):

        self.hidden_layers = hidden_layers
        self.activations = activations
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epoch
        self.regularizer = regularizer
        self.regularizer_scale = regularizer_scale
        self.batch_normalization = batch_normalization
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.model_name = datetime.now().strftime('model_%Y%m%d-%H%M%S')

        if save_path is not None:
            while Path(self.save_path +
                               '/' + self.model_name + '.ckpt').exists():
                self.model_name = self.model_name + "_"

        # network structure
        if hidden_layers is None:
            self.hidden_layers = [100]
        if activations is None:
            self.activations = ["relu"]

        # exceptions
        if len(self.hidden_layers) != len(self.activations):
            assert "length of hidden layers and activation is not same."
        if optimizer != "Adam" and optimizer != "GradientDescent":
            assert "invalid optimizer"

    def model(self, X_train, y_train=None):
        n_samples, n_features = np.shape(X_train)
        _, n_classes = np.shape(y_train)

        if y_train is None:
            # TODO
            n_classes = 4

        with tf.variable_scope("network"):
            # input
            X_tf = tf.placeholder(tf.float32,
                                  shape=[None, n_features],
                                  name='X')
            y_tf = tf.placeholder(tf.float32,
                                  shape=[None, n_classes],
                                  name='y')
            is_training_tf = tf.placeholder(tf.bool,
                                            name='is_training')

            # build graph
            net = X_tf
            for i, (hidden_layer, activation) \
                    in enumerate(zip(self.hidden_layers, self.activations)):
                with tf.variable_scope('layer{}'.format(i)):
                    # xavier initialization for variables
                    xavier_initializer = tf.contrib.layers.xavier_initializer()

                    # activation function
                    if activation == 'relu':
                        activation_fn = tf.nn.relu
                    elif activation == 'sigmoid':
                        activation_fn = tf.nn.sigmoid
                    elif activation == 'elu':
                        activation_fn = tf.nn.elu
                    else:
                        activation_fn = tf.nn.relu

                    # regularizer
                    if self.regularizer == 'l1':
                        regularizer = l1_regularizer(self.regularizer_scale)
                    elif self.regularizer == 'l2':
                        regularizer = l2_regularizer(self.regularizer_scale)
                    else:
                        regularizer = l2_regularizer(self.regularizer_scale)

                    # fully connected graph
                    net = fully_connected(
                        net, hidden_layer,
                        activation_fn=activation_fn,
                        biases_initializer=xavier_initializer,
                        weights_initializer=xavier_initializer,
                        weights_regularizer=regularizer)

                    # batch normalization
                    if self.batch_normalization:
                        net = tf.layers. \
                            batch_normalization(net,
                                                training=is_training_tf)

                    # dropout
                    if self.dropout:
                        net = dropout(net,
                                      keep_prob=1-self.dropout_rate,
                                      is_training=is_training_tf)

            # end of build graph
            net = flatten(net)
            net = tf.layers.dense(net, n_classes)

        return net, X_tf, y_tf, is_training_tf

    def batches(self, X_train, y_train):
        # size
        n_samples, _ = np.shape(X_train)
        batch_size = self.batch_size

        # batchs
        batches = []

        random_mask = list(np.random.permutation(n_samples))
        random_X = X_train[random_mask, :]
        random_Y = y_train[random_mask, :]

        num_batches = n_samples // batch_size
        for i in range(0, num_batches):
            batch_X = random_X[batch_size * i:batch_size * i + batch_size, :]
            batch_Y = random_Y[batch_size * i:batch_size * i + batch_size, :]
            batch = (batch_X, batch_Y)
            batches.append(batch)

        if n_samples % batch_size != 0:
            batch_X = random_X[num_batches * batch_size:n_samples, :]
            batch_Y = random_Y[num_batches * batch_size:n_samples, :]
            batch = (batch_X, batch_Y)
            batches.append(batch)

        return batches

    def fit(self, X, y, sample_weight=None):

        print("------------------------------------")
        print("NeuralNetClassifier fit")

        # size
        n_samples, n_features = np.shape(X)
        _, n_classes = np.shape(y)

        # generate batches
        batches = self.batches(X_train=X, y_train=y)

        # build neural net
        network, X_tf, y_tf, is_training_tf = self.model(X, y)

        # cost (loss)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_tf,
                                                    logits=network))

        # optimizer
        if self.optimizer == 'Adam':
            optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'GradientDescent':
            optimizer = \
                tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate)
        else:
            optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # When using the batchnormalization layers,
        # it is necessary to manually add the update operations
        # because the moving averages are not included in the graph
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="network")

        with tf.control_dependencies(update_op):
            train_op = optimizer.minimize(loss)

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # tensorflow seesion
        with tf.Session() as sess:

            # initialization
            sess.run(init_op)

            for epoch in range(self.num_epochs):
                for batch in batches:
                    batch_X, batch_y = batch

                    feed = {
                        X_tf: batch_X,
                        y_tf: batch_y,
                        is_training_tf: True
                    }

                    _, loss_val = sess.run([train_op, loss],
                                           feed_dict=feed)

                    if (epoch % 100) == 0:
                        print(epoch, loss_val)

            if self.save_path is not None and self.model_name is not None:
                save_path = self.save_path + '/' \
                            + self.model_name + '.cpkl'
                saved_path = saver.save(sess, save_path)
                print("fitted model save: {}".format(saved_path))

        return self

    def predict_proba(self, X):

        print("------------------------------------")
        print("NeuralNetClassifier predict_proba")

        # size
        n_samples, n_features = np.shape(X)
        n_classes = 4 # TODO

        # build neural net
        network, X_tf, _, is_training_tf = self.model(X)

        # tensorflow seesion
        saver = tf.train.Saver()

        with tf.Session() as sess:
            save_path = self.save_path + '/' \
                        + self.model_name + '.cpkl'
            saver.restore(sess, save_path)
            print("fitted model restored: {}".format(save_path))

            feed = {
                X_tf: X,
                is_training_tf: False
            }

            predict_op = tf.nn.softmax(network, name='softmax')
            P_predicted = sess.run(predict_op, feed_dict=feed)

        return P_predicted

    def score(self, X, y, sample_weight=None):
        P_predicted = self.predict_proba(X)
        n_samples, n_labels = np.shape(P_predicted)

        score = np.zeros(n_samples)

        for i in range(0, n_samples):
            score[i] = spearmanr(y[i, :], P_predicted[i, :])[0]

        return np.mean(score)