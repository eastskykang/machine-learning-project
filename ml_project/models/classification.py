import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from scipy.stats import spearmanr
from datetime import datetime
from pathlib import Path
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.preprocessing import sequence


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
            GradientDescentOptimizer(learning_rate=self.learning_rate). \
            minimize(cost)

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


class ConvolutionalNeuralNetClassifier(BaseEstimator, TransformerMixin):
    """Convolutional Neural Net Classifier"""
    def __init__(self,
                 batch_size=128, dropout=False, dropout_rate=0.3,
                 optimizer='Adam', learning_rate=0.001, num_epoch=50,
                 save_path=None, verbosity=1):

        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.save_path = save_path
        self.model_name = datetime.now().strftime('model_%Y%m%d-%H%M%S')
        self.model_path = None
        self.one_hot_encoder = None

        # training / evaluation mask
        self.training_mask = None
        self.evaluation_mask = None

        # exceptions
        if optimizer != "Adam" and optimizer != "GradientDescent":
            assert "invalid optimizer"

    def model(self, X_train):
        n_samples, n_features = np.shape(X_train)

        tf.reset_default_graph()
        with tf.variable_scope("network"):
            # input
            X_tf = tf.placeholder(tf.float32,
                                  shape=[None, n_features],
                                  name='X')
            y_tf = tf.placeholder(tf.float32,
                                  name='y')
            is_training_tf = tf.placeholder(tf.bool,
                                            name='is_training')

            # build graph
            net = tf.expand_dims(X_tf, axis=-1)

            with tf.variable_scope('layers'):
                # cnn 1
                net = tf.layers.conv1d(
                    net,
                    filters=8,
                    kernel_size=256,
                    strides=1,
                    activation=tf.nn.relu)

                # max pooling 1
                net = tf.layers.max_pooling1d(
                    net,
                    pool_size=2,
                    strides=1)

                # cnn 2
                net = tf.layers.conv1d(
                    net,
                    filters=8,
                    kernel_size=128,
                    strides=1,
                    activation=tf.nn.relu)

                # max pooling 2
                net = tf.layers.max_pooling1d(
                    net,
                    pool_size=4,
                    strides=2)

                # flattening
                net = tf.contrib.layers.flatten(net)

                # dense layer 1
                net = tf.layers.dense(
                    net,
                    units=1024,
                    activation=tf.nn.relu)

                net = tf.layers.dropout(inputs=net,
                                        rate=0.3,
                                        training=is_training_tf)

                # dense layer 2
                net = tf.layers.dense(
                    net,
                    units=32,
                    activation=tf.nn.relu)

                net = tf.layers.dropout(inputs=net,
                                        rate=0.3,
                                        training=is_training_tf)

                # logit
                logits_tf = tf.layers.dense(
                    net,
                    units=4,
                    activation=None)

                # probs
                probs_tf = tf.nn.softmax(logits_tf)

                # prediction
                predictions_tf = tf.argmax(probs_tf, axis=1)

        return X_tf, y_tf, is_training_tf, logits_tf, probs_tf, predictions_tf

    def random_batches(self, X_train, y_train):

        print("         generate batch")

        # size
        n_samples, _ = np.shape(X_train)
        batch_size = self.batch_size

        # batchs
        batches = []

        random_mask = list(np.random.permutation(n_samples))
        random_X = X_train[random_mask, :]
        random_Y = y_train[random_mask]

        num_batches = n_samples // batch_size
        for i in range(0, num_batches):
            batch_X = random_X[batch_size * i:batch_size * i + batch_size, :]
            batch_Y = random_Y[batch_size * i:batch_size * i + batch_size]
            batch = (batch_X, batch_Y)
            batches.append(batch)

        if n_samples % batch_size != 0:
            batch_X = random_X[num_batches * batch_size:n_samples, :]
            batch_Y = random_Y[num_batches * batch_size:n_samples]
            batch = (batch_X, batch_Y)
            batches.append(batch)

        return batches

    def order_batches(self, X_train, y_train):
        # size
        n_samples, _ = np.shape(X_train)
        batch_size = self.batch_size

        # batchs
        batches = []

        num_batches = n_samples // batch_size
        for i in range(0, num_batches):
            batch_X = X_train[batch_size * i:batch_size * i + batch_size, :]
            batch_Y = y_train[batch_size * i:batch_size * i + batch_size]
            batch = (batch_X, batch_Y)
            batches.append(batch)

        if n_samples % batch_size != 0:
            batch_X = X_train[num_batches * batch_size:n_samples, :]
            batch_Y = y_train[num_batches * batch_size:n_samples]
            batch = (batch_X, batch_Y)
            batches.append(batch)

        return batches

    def fit(self, X, y, sample_weight=None):

        print("------------------------------------")
        print("CNNClassifier fit")

        # training / evaluation data
        n_samples, n_features = np.shape(X)

        mask = list(np.random.permutation(n_samples))
        self.training_mask = mask[0:6200]
        self.evaluation_mask = mask[6200:None]
        X_eval = X[self.evaluation_mask, :]
        y_eval = y[self.evaluation_mask]

        # build network
        X_tf, y_tf, is_training_tf, \
        logits, probs_tf, prediction_tf = \
            self.model(X[self.training_mask, :])

        # cost (loss)
        labels = tf.cast(y_tf, tf.int32)
        labels = tf.one_hot(labels, depth=4)

        labels = tf.cast(labels, tf.float64)
        logits = tf.cast(logits, tf.float64)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                    logits=logits))

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

        # train operation
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())

        # initialization operation
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # tensorflow session
        with tf.Session() as sess:

            # initialization run
            print("     initialization")
            sess.run(init_op)

            # training
            print("     training")

            for epoch in range(self.num_epochs):

                # =============================================================
                # training with batch
                batches = self.random_batches(X_train=X[self.training_mask, :],
                                              y_train=y[self.training_mask])

                for (batch_X, batch_y) in batches:

                    feed_train = {
                        X_tf: batch_X,
                        y_tf: batch_y.astype('float32'),
                        is_training_tf: True
                    }

                    _, loss_train, prediction_train = \
                        sess.run([train_op, loss, prediction_tf],
                                 feed_dict=feed_train)

                    # training score
                    score_train = f1_score(batch_y,
                                           prediction_train,
                                           average="micro")

                    # evaluation score
                    feed_eval = {
                        X_tf: X_eval,
                        y_tf: y_eval.astype('float32'),
                        is_training_tf: False
                    }

                    loss_eval, prediction_eval = sess.run([loss, prediction_tf],
                                                          feed_dict=feed_eval)
                    score_eval = f1_score(y_eval,
                                          prediction_eval,
                                          average="micro")

                    print("         epoch = {} / "
                          "loss train   = {} / "
                          "f1 train     = {} / "
                          "loss eval    = {} / "
                          "f1 eval      = {}".format(epoch,
                                                     loss_train,
                                                     score_train,
                                                     loss_eval,
                                                     score_eval))

            # save tensorflow model
            if self.save_path is None:
                self.save_path = 'data/tmp/'

            # model path
            while Path(self.save_path + self.model_name).exists():
                self.model_name = self.model_name + "_"

            # create directory
            Path(self.save_path + self.model_name). \
                mkdir(exist_ok=False, parents=True)
            self.model_path = \
                self.save_path + self.model_name + '/model.ckpt'

            # save model
            tf_save_path = self.model_path
            tf_saved_path = saver.save(sess, tf_save_path)
            print("fitted model save: {}".format(tf_saved_path))
            # print("f1 score: {}".format(self.score(X, y)))

        return self

    def predict(self, X):
        print("------------------------------------")
        print("CNNClassifier predict")

        # build neural net
        X_tf, _, is_training_tf, _, _, predictions_tf = self.model(X)

        # tensorflow seesion
        saver = tf.train.Saver()

        with tf.Session() as sess:
            save_path = self.model_path
            saver.restore(sess, save_path)
            print("fitted model restored: {}".format(save_path))

            feed = {
                X_tf: X,
                is_training_tf: False
            }

            prediction = sess.run(predictions_tf, feed_dict=feed)

        return prediction

    def score(self, X, y, sample_weight=None):
        y_predicted = self.predict(X)
        score = f1_score(y, y_predicted, average="micro")
        print("f1 score = {}".format(score))

        return score

    def set_save_path(self, save_path):
        self.save_path = save_path


class LSTMClassifier(BaseEstimator, TransformerMixin):
    """LSTM Classifier for sequential data"""
    def __init__(self, dropout_rate=0.3, save_path=None, lstm_layers=None,
                 batch_size=100, num_epoch=300, max_len=500, n_feature=1,
                 optimizer='adam'):

        self.dropout_rate = dropout_rate
        self.lstm_layers = lstm_layers
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.save_path = save_path
        self.max_len = max_len
        self.n_feature = n_feature
        self.optimizer = optimizer

        self.model_name = datetime.now().strftime('model_%Y%m%d-%H%M%S')
        self.model_path = None

        # network structure
        if lstm_layers is None:
            self.lstm_layers = [8]

    def model(self, timestep, n_feature):

        # model
        model = Sequential()

        # lstm
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i is 0 and i is len(self.lstm_layers) - 1:
                # first layer also last layer
                model.add(LSTM(lstm_layer, input_shape=(timestep, n_feature), dropout=self.dropout_rate))
            elif i is 0 and i is not len(self.lstm_layers) - 1:
                # first layer
                model.add(LSTM(lstm_layer, input_shape=(timestep, n_feature), return_sequences=True, dropout=self.dropout_rate))
            elif i is len(self.lstm_layers) - 1:
                # last layer
                model.add(LSTM(lstm_layer, dropout=self.dropout_rate))
            else:
                model.add(LSTM(lstm_layer, return_sequences=True, dropout=self.dropout_rate))

        # output
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['accuracy'])

        print("LSTMClassifier model")
        print(model.summary())

        return model

    def fit(self, X, y, sample_weight=None):

        print("------------------------------------")
        print("LSTMClassifier fit")

        # truncate X
        X = sequence.pad_sequences(X, maxlen=self.max_len * self.n_feature, truncating="post")
        n_samples, n_timestep = np.shape(X)

        # kth order (feature)
        timestep = int(n_timestep / self.n_feature)
        X = np.reshape(X, (n_samples, timestep, self.n_feature))

        print("input shape = {}".format(np.shape(X)))

        # class weight (for imbalance data)
        class_weight = compute_class_weight('balanced', np.unique(y), y)
        sample_weight = compute_sample_weight('balanced', y)

        # one hot encoding
        one_hot_encoder = LabelBinarizer()
        one_hot_encoder.fit(y)
        y = one_hot_encoder.transform(y)

        # model
        # call back for early stopping
        callback = [
            EarlyStopping(monitor='loss', min_delta=1e-4, verbose=1)
        ]

        # network
        net = self.model(timestep, self.n_feature)
        net.fit(X.astype(float), y,
                epochs=self.num_epoch,
                batch_size=self.batch_size,
                # callbacks=callback,
                sample_weight=sample_weight,
                verbose=2)

        # model path
        if self.save_path is None:
            self.save_path = 'data/tmp/'

        while Path(self.save_path + self.model_name).exists():
            self.model_name = self.model_name + "_"

        # create directory
        Path(self.save_path + self.model_name). \
            mkdir(exist_ok=False, parents=True)
        self.model_path = \
            self.save_path + self.model_name + '/model.h5'

        # save model
        net.save(self.model_path)
        print("fitted model save: {}".format(self.model_path))
        del net

        return self

    def predict_proba(self, X):
        print("------------------------------------")
        print("LSTMClassifier predict_proba")

        # load model
        net = load_model(self.model_path)

        # truncate X
        X = sequence.pad_sequences(X, maxlen=self.max_len * self.n_feature,
                                   truncating="post")
        n_samples, n_timestep = np.shape(X)

        # kth order (feature)
        timestep = int(n_timestep / self.n_feature)
        X = np.reshape(X, (n_samples, timestep, self.n_feature))

        return net.predict(X.astype(float))

    def predict(self, X):
        print("------------------------------------")
        print("LSTMClassifier predict")

        P_predicted = self.predict_proba(X)
        return np.argmax(P_predicted, axis=1)

    def score(self, X, y, sample_weight=None):
        y_predicted = self.predict(X)
        return f1_score(y, y_predicted)


class NeuralNetClassifier(BaseEstimator, TransformerMixin):
    """Neural Net Classifier"""
    def __init__(self, hidden_layers=None, activations=None, regularizer='l2',
                 regularizer_scale=1.0, batch_normalization=True,
                 batch_size=58, dropout=True, dropout_rate=0.3,
                 optimizer='Adam', learning_rate=0.01, num_epoch=500,
                 score_metric='f1', one_hot_encoding=True, weighted_class=True,
                 save_path=None):

        self.hidden_layers = hidden_layers
        self.activations = activations
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epoch
        self.regularizer = regularizer
        self.regularizer_scale = regularizer_scale
        self.batch_normalization = batch_normalization
        self.batch_size = batch_size
        self.score_metric = score_metric
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.one_hot_encoding = one_hot_encoding
        self.weighted_class = weighted_class
        self.save_path = save_path
        self.model_name = datetime.now().strftime('model_%Y%m%d-%H%M%S')
        self.model_path = None
        self.one_hot_encoder = None

        # network structure
        if hidden_layers is None:
            self.hidden_layers = [128, 32]
        if activations is None:
            self.activations = ['relu', 'relu']

        # exceptions
        if len(self.hidden_layers) != len(self.activations):
            assert "length of hidden layers and activation is not same."
        if optimizer != "Adam" and optimizer != "GradientDescent":
            assert "invalid optimizer"

    def model(self, X_train, y_train=None):
        n_samples, n_features = np.shape(X_train)

        if y_train is None:
            n_classes = 4
        else:
            _, n_classes = np.shape(y_train)

        tf.reset_default_graph()
        with tf.variable_scope("network"):
            # input
            X_tf = tf.placeholder(tf.float64,
                                  shape=[None, n_features],
                                  name='X')
            y_tf = tf.placeholder(tf.float64,
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
                        regularizer = tf.contrib.layers.l1_regularizer(self.regularizer_scale)
                    elif self.regularizer == 'l2':
                        regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_scale)
                    else:
                        regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_scale)

                    # fully connected graph
                    net = tf.contrib.layers.fully_connected(
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
                        net = tf.contrib.layers.dropout(net,
                                                        keep_prob=(1-self.dropout_rate),
                                                        is_training=is_training_tf)

            # end of build graph
            net = tf.contrib.layers.flatten(net)
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

        # class weight
        if self.weighted_class:
            class_weight = compute_class_weight('balanced', np.unique(y), y)

        # one hot encoder
        if self.one_hot_encoding:
            self.one_hot_encoder = LabelBinarizer()
            self.one_hot_encoder.fit(y)
            y_onehot = self.one_hot_encoder.transform(y)

        _, n_classes = np.shape(y_onehot)

        # generate batches
        batches = self.batches(X_train=X, y_train=y_onehot)

        # build neural net
        network, X_tf, y_tf, is_training_tf = self.model(X, y_onehot)

        # cost (loss)
        if self.weighted_class:
            weight_class = tf.reshape(class_weight, [4, 1])
            weight_per_sample = tf.matmul(y_tf, weight_class)

            # cost (class weighted)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=y_tf,
                logits=network)
            loss = tf.multiply(weight_per_sample, loss)
            loss = tf.reduce_mean(loss)
        else:
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

            # save tensorflow model
            if self.save_path is None:
                self.save_path = 'data/tmp/'

            # model path
            while Path(self.save_path + self.model_name).exists():
                self.model_name = self.model_name + "_"

            # create directory
            Path(self.save_path + self.model_name). \
                mkdir(exist_ok=False, parents=True)
            self.model_path = \
                self.save_path + self.model_name + '/model.ckpt'

            # save model
            tf_save_path = self.model_path
            tf_saved_path = saver.save(sess, tf_save_path)
            print("fitted model save: {}".format(tf_saved_path))

        if self.score_metric is 'f1':
            print("f1 score: {}".format(self.score(X, y)))

        return self

    def predict_proba(self, X):

        print("------------------------------------")
        print("NeuralNetClassifier predict_proba")

        # build neural net
        network, X_tf, _, is_training_tf = self.model(X)

        # tensorflow seesion
        saver = tf.train.Saver()

        with tf.Session() as sess:
            save_path = self.model_path
            saver.restore(sess, save_path)
            print("fitted model restored: {}".format(save_path))

            feed = {
                X_tf: X,
                is_training_tf: False
            }

            predict_op = tf.nn.softmax(network, name='softmax')
            P_predicted = sess.run(predict_op, feed_dict=feed)

        return P_predicted

    def predict(self, X):
        print("------------------------------------")
        print("NeuralNetClassifier predict")

        P_predicted = self.predict_proba(X)
        return np.argmax(P_predicted, axis=1)

    def score(self, X, y, sample_weight=None):
        if self.score_metric is 'spearmanr':
            P_predicted = self.predict_proba(X)
            n_samples, n_labels = np.shape(P_predicted)
            score = np.zeros(n_samples)

            for i in range(0, n_samples):
                score[i] = spearmanr(y[i, :], P_predicted[i, :])[0]

            score = np.mean(score)
        elif self.score_metric is 'f1':
            y_predicted = self.predict(X)
            score = f1_score(y, y_predicted, average="micro")
        else:
            y_predicted = self.predict(X)
            score = f1_score(y, y_predicted, average="micro")

        return score

    def set_save_path(self, save_path):
        self.save_path = save_path