# MIT License

# Copyright (c) 2020 Khaled Zaouk

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Original Paper (Rifai et al.):
# https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf

import tensorflow as tf
from saver import Saver
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import numpy as np
import logging
import time


def weight_variable(shape, trainable=True, init_std=0.1):
    # tensorflow 2.0; 1.x counterpart is tf.truncated_normal
    initial = tf.random.truncated_normal(shape, stddev=init_std)
    return tf.Variable(initial, trainable=trainable)


def bias_variable(shape, trainable=True, init=0.1):
    initial = tf.constant(init, shape=shape)
    return tf.Variable(initial, trainable=trainable)


class CAE:
    def __init__(self, input_dim, layer_sizes, activations, lamda=1e-1,
                 learning_rate=0.001, batch_size=128, n_epochs=100,
                 early_stopping=True, patience=10, random_state=42):
        """
        Implements a contractive auto-encoder

        input_dim: int
            Dimension of the input vector
        layer_sizes: list
            List of integers concerning the size of the layers in the encoder
            part.
        activations: list
            List of str with activations up to the encoding layer (included)
        lamda: float
            Coefficient that multiplies the jacobian term of the loss function.
        learning_rate: float
            Learning rate used for updating weights when minimizing the loss
            function.
        batch_size: int
            Size of the batch on which one training step is done
        n_epochs: int
            Number of epochs for training
        early_stopping: boolean
            Whether to use early stopping mechanism to stop training and avoid
            overfitting.
        patience: int
            Number of epochs to watch before stopping the training. Only
            useful if early_stopping is True.
        random_state: int
            Seed to be used with pseudo-random number generator.

        """
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.session = tf.compat.v1.Session()
        self.best_epoch_ = None
        self.last_fit_duration_ = None
        self.lamda = lamda

        # no mixing of different types of activations
        assert len(set(activations)) == 1
        assert len(layer_sizes) <= 2

    def get_jacobian_loss(self):
        encodings = self.forward_pass(self.input_pl)
        if len(self.layer_sizes) == 1 and self.activations[0] == 'sigmoid':
            w = self.weights[0]
            w_sum_over_input_dim = tf.reduce_sum(tf.square(w), axis=0)
            w_ = tf.expand_dims(w_sum_over_input_dim, 1)
            h_ = tf.square(encodings * (1 - encodings))
            h_times_w_ = tf.matmul(h_, w_)
            jacobian = tf.reduce_mean(h_times_w_)

        elif len(self.layer_sizes) == 1 and self.activations[0] == 'relu':
            w = self.weights[0]
            b = self.biases[0]
            pre_activation = tf.matmul(
                self.input_pl, w) + b
            indicator = tf.nn.relu(tf.sign(pre_activation))
            w_s = tf.square(w)
            w_ = tf.transpose(tf.reduce_sum(w_s, axis=0, keepdims=True))
            batch_jacobian_vec = tf.matmul(indicator, w_)
            jacobian = tf.reduce_mean(batch_jacobian_vec)

        elif len(self.layer_sizes) == 2 and self.activations[0] == 'sigmoid':
            w1_var = self.weights[0]
            w2_var = self.weights[1]
            b1_var = self.biases[0]
            x_pl = self.input_pl
            intermediate = tf.nn.sigmoid(tf.matmul(x_pl, w1_var) + b1_var)
            z_ = intermediate * (1 - intermediate)
            aux = tf.expand_dims(z_, 2) * w2_var
            k_sum = tf.matmul(w1_var, aux)
            k_ss = tf.square(k_sum)
            sum_k_ss = tf.reduce_sum(k_ss, axis=1)
            h_ = tf.square(encodings * (1 - encodings))
            batch_jacobian_vec = tf.reduce_sum(h_ * sum_k_ss, axis=1)
            jacobian = tf.reduce_mean(batch_jacobian_vec)
        elif len(self.layer_sizes) == 2 and self.activations[0] == 'relu':
            x_pl = self.input_pl
            w1_var = self.weights[0]
            w2_var = self.weights[1]
            b1_var = self.biases[0]
            b2_var = self.biases[1]
            preac_1 = tf.matmul(x_pl, w1_var) + b1_var
            intermediate = tf.nn.relu(preac_1)
            indicator_1 = tf.nn.relu(tf.sign(preac_1))
            preac_2 = tf.matmul(intermediate, w2_var) + b2_var
            indicator_2 = tf.nn.relu(tf.sign(preac_2))
            z_ = indicator_1
            aux = tf.expand_dims(z_, 2) * w2_var
            k_sum = tf.matmul(w1_var, aux)
            k_ss = tf.square(k_sum)
            sum_k_ss = tf.reduce_sum(k_ss, axis=1)
            h_ = indicator_2
            batch_jacobian_vec = tf.reduce_sum(h_ * sum_k_ss, axis=1)
            jacobian = tf.reduce_mean(batch_jacobian_vec)
        else:
            raise NotImplementedError(
                "Jacobian not yet implemented for this activation: {}".format(
                    self.activations[0]))
        return jacobian

    def compile(self):
        self._init_architecture()
        obs_approx = self.full_forward_pass(self.input_pl)
        encodings = self.forward_pass(self.input_pl)

        # recons_loss:
        recons_loss = tf.reduce_mean(tf.square(obs_approx - self.input_pl))
        jacobian_loss = self.get_jacobian_loss()
        loss = recons_loss + self.lamda * jacobian_loss

        train_step = tf.compat.v1.train.AdamOptimizer(
            self.learning_rate).minimize(loss)

        self.loss = loss
        self.recons_loss = recons_loss
        self.jacobian_loss = jacobian_loss
        self.train_step = train_step

    def forward_pass(self, input_pl):
        # forward pass only through the encoder layer
        output = input_pl
        for i in range(len(self.weights)):
            output = tf.matmul(output, self.weights[i]) + self.biases[i]
            if self.activations[i] == 'relu':
                output = tf.nn.relu(output)
            elif self.activations[i] == 'sigmoid':
                output = tf.nn.sigmoid(output)
            elif self.activations[i] == '' or self.activations[i] is None:
                pass
            else:
                raise NotImplementedError(
                    "This activation ({}) is not yet implemented.".format(
                        self.activations[i]))
        return output

    def full_forward_pass(self, input_pl):
        encoding = self.forward_pass(input_pl)
        output = encoding
        for i in range(len(self.decoder_weights)):
            output = tf.matmul(
                output, self.decoder_weights[i]) + self.decoder_biases[i]
            if self.activations[len(self.activations) - i - 1] == 'relu':
                output = tf.nn.relu(output)
        return output

    def _init_architecture(self):
        tf.compat.v1.disable_eager_execution()
        self.input_pl = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.input_dim))

        weights = []
        biases = []
        i_dim = self.input_dim

        # encoder layers
        for layer_size in self.layer_sizes:
            w = weight_variable([i_dim, layer_size])
            b = bias_variable([layer_size])
            i_dim = layer_size
            weights.append(w)
            biases.append(b)

        # decoder layers (tied weights case)
        decoder_weights = []
        decoder_biases = []
        for w in weights[::-1]:
            decoder_weights.append(tf.transpose(w))
            decoder_biases.append(bias_variable([int(w.shape[0])]))

        self.weights = weights
        self.biases = biases
        self.decoder_weights = decoder_weights
        self.decoder_biases = decoder_biases

        self.saver = Saver(self.weights + self.biases + self.decoder_biases)

    def get_fd(self, X):
        return {
            self.input_pl: X,
        }

    def eval_var(self, var, X):
        return var.eval(
            feed_dict=self.get_fd(X),
            session=self.session)

    def log_losses(self, X, val=False, verbose=False, e=0):
        recons_loss = self.eval_var(self.recons_loss, X)
        jacobian_loss = self.eval_var(self.jacobian_loss, X)
        loss = recons_loss + self.lamda * jacobian_loss

        if not val:
            self.history['loss'].append(loss)
            self.history['recons_loss'].append(recons_loss)
            self.history['jacobian_loss'].append(jacobian_loss)
        else:
            self.history['val_loss'].append(loss)
            self.history['val_recons_loss'].append(recons_loss)
            self.history['val_jacobian_loss'].append(jacobian_loss)

        if verbose:
            if val:
                prefix = "[VAL]"
            else:
                prefix = "[TRAIN]"
            logging.info("{} Epoch {} - Losses: recons: {:.5f} \t jacobi: {:.5f} \t total: {:.5f}".format(
                prefix, e,  recons_loss, jacobian_loss, loss))
        return loss

    def log_losses_batch(self, batches, val=False, verbose=False, e=0):
        # batch version of calculating the losses (by batch then avg
        # across batches)
        recons_loss = 0
        jacobian_loss = 0
        count = 0
        for key in batches:
            batch = batches[key]
            recons_loss += self.eval_var(self.recons_loss, batch)
            jacobian_loss += self.eval_var(self.jacobian_loss, batch)
            count += 1

        recons_loss /= count
        jacobian_loss /= count
        loss = recons_loss + self.lamda * jacobian_loss

        if not val:
            self.history['loss'].append(loss)
            self.history['recons_loss'].append(recons_loss)
            self.history['jacobian_loss'].append(jacobian_loss)
        else:
            self.history['val_loss'].append(loss)
            self.history['val_recons_loss'].append(recons_loss)
            self.history['val_jacobian_loss'].append(jacobian_loss)

        if verbose:
            if val:
                prefix = "[VAL]"
            else:
                prefix = "[TRAIN]"
            logging.info("{} Epoch {} - Losses: recons: {:.5f} \t jacobi: {:.5f} \t total: {:.5f}".format(
                prefix, e,  recons_loss, jacobian_loss, loss))
        return loss

    def fit(self, X, log_time=False):
        t0 = time.time()

        self.history = {
            'loss': [],
            'recons_loss': [],
            'jacobian_loss': []}
        if self.early_stopping:
            X, X_val = train_test_split(X, shuffle=False)
            self.history['val_loss'] = []
            self.history['val_recons_loss'] = []
            self.history['val_jacobian_loss'] = []

        n_points = len(X)
        sess = self.session
        sess.run(tf.compat.v1.global_variables_initializer())
        self.log_losses(X, verbose=True)

        if self.early_stopping:
            self.log_losses(X_val, val=True, verbose=True)

        n_batches = int(np.ceil(n_points / self.batch_size))
        bs = self.batch_size

        best_epoch = -1
        min_err = np.inf

        logging.info("Initial loss(es): {}".format(self.history))
        for e in range(self.n_epochs):
            if self.early_stopping and best_epoch > 0 and e > best_epoch + self.patience:
                exited_early_stopping = True
                break

            X = shuffle(X, random_state=self.random_state+e)
            for i in range(n_batches):
                if (i % 100) == 0:
                    logging.info(
                        "Epoch: {} \t step: {}/{} batches".format(e, i, n_batches))
                x_batch = X[i*bs:(i+1)*bs, :]
                self.train_step.run(feed_dict=self.get_fd(x_batch),
                                    session=self.session)
            loss_value = self.log_losses(X, verbose=True, e=e)
            if self.early_stopping:
                val_loss_value = self.log_losses(
                    X_val, val=True, verbose=True, e=e)
                if val_loss_value < min_err:
                    min_err = val_loss_value
                    best_epoch = e
                    self.best_epoch_ = e
                    self.saver.save_weights(self.session)
                    logging.info("===> Epoch: {} \t loss: {:.6f} \t val_loss: {:.6f} ** (new best epoch)".format(
                        e, loss_value, val_loss_value))

        if self.early_stopping:
            self.saver.restore_weights(self.session)
        else:
            self.saver.save_weights(self.session)

        tend = time.time()
        fitting_time = tend - t0
        self.last_fit_duration_ = fitting_time

        if log_time:
            logging.info(
                "[autoencoder fitting time]: {} minutes and {} seconds".format(
                    fitting_time // 60,
                    int(fitting_time % 60)))
        return self.history

    def transform(self, X):
        output_var = self.forward_pass(self.input_pl)
        output = output_var.eval(
            feed_dict={self.input_pl: X},
            session=self.session)
        return output

    def persist(self, fpath):
        data = self.get_persist_info()
        if os.path.dirname(fpath) != "":
            if not os.path.exists(os.path.dirname(fpath)):
                os.path.makedirs(os.path.dirname(fpath))
        np.save(fpath, data)

    def serialize(self, fpath):
        self.persist(fpath)

    def get_persist_info(self):
        signature_data = {
            'input_dim': self.input_dim,
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'random_state': self.random_state,
            'lamda': self.lamda
        }
        other_data = {
            'best_weights': self.saver.best_params,  # ws and bs
            'history': self.history,
            'best_epoch': self.best_epoch_,
            'last_fit_duration': self.last_fit_duration_
        }
        return {'signature': signature_data,
                'other': other_data}

    def clone(self):
        data = self.get_persist_info()
        return CAE.make_instance(data['signature'], data['other'])

    @staticmethod
    def make_instance(signature_data, other_data):
        instance = CAE(**signature_data)
        instance.compile()
        instance.saver.best_params = other_data['best_weights'].copy()
        instance.saver.restore_weights(instance.session)
        instance.history = other_data['history'].copy()
        instance.last_fit_duration_ = other_data['last_fit_duration']
        instance.best_epoch_ = other_data['best_epoch']
        return instance

    @staticmethod
    def load_from_file(fpath):
        data = np.load(fpath, allow_pickle=True)[()]
        return CAE.make_instance(data['signature'],
                                 data['other'])
