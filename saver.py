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

import tensorflow as tf
import numpy as np


class Saver:
    """A Saver class that keeps track of the best weights, biases and embedding
    matrix.
    During training, update_weights is called whenever the validation error
    decreases. After training has finished, we can call get_weights to get
    the weights that yielded the smallest error on the validation set.
    The sole purpose of this class is to avoid overfitting the neural network
    and do early stopping.
    """

    def __init__(self, params):
        # Maintain a reference for the tensor Variables.
        self.params = params
        self.best_params = []

    def save_weights(self, sess):
        """
        # read current weights from session and store them in best params
        # read from session
        """
        self.best_params = []
        for param in self.params:
            self.best_params.append(param.eval(sess))

    def restore_weights(self, sess):
        """
        # put best params in the session weights
        # write to session
        """
        if len(self.best_params) > 0:
            i = 0
            for param in self.params:
                sess.run(param.assign(self.best_params[i]))
                i += 1
        else:
            print("[Logger: No best parameters found]")

    def get_weights(self):
        return self.best_params
