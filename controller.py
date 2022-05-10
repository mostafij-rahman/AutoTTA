import PIL.Image
import numpy as np

import tensorflow as tf
from keras import models, layers, backend, initializers
from keras.layers import Dense, Activation

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
backend.set_session(session)

from transformations import get_transformations


# Experiment parameters

LSTM_UNITS = 100

OP_TYPES = 15
OP_PROBS = 11
OP_MAGNITUDES = 10


transformations = get_transformations()

class Operation:
    def __init__(self, types_softmax, probs_softmax, magnitudes_softmax, argmax=False):
        if argmax:
            self.type = types_softmax.argmax()
            t = transformations[self.type]
            self.prob = probs_softmax.argmax() / (OP_PROBS-1)
            m = magnitudes_softmax.argmax() / (OP_MAGNITUDES-1)
            self.magnitude = m*(t[2]-t[1]) + t[1]
        else:
            self.type = np.random.choice(OP_TYPES, p=types_softmax)
            t = transformations[self.type]
            self.prob = np.random.choice(np.linspace(0, 1, OP_PROBS), p=probs_softmax)
            self.magnitude = np.random.choice(np.linspace(t[1], t[2], OP_MAGNITUDES), p=magnitudes_softmax)
        self.transformation = t[0]

    def __call__(self, X):
        _X = []
        for x in X:
            if np.random.rand() < self.prob:
                x = PIL.Image.fromarray(x)
                x = self.transformation(x, self.magnitude)
            _X.append(np.array(x))
        return np.array(_X)

    def __str__(self):
        return 'Operation %2d (P=%.3f, M=%.3f)' % (self.type, self.prob, self.magnitude)

class Subpolicy:
    def __init__(self, *operations):
        self.operations = operations

    def __call__(self, X):
        for op in self.operations:
            X = op(X)
        return X

    def __str__(self):
        ret = ''
        for i, op in enumerate(self.operations):
            ret += str(op)
            if i < len(self.operations)-1:
                ret += '\n'
        return ret

class Controller:
    def __init__(self, n_subpolicies = 5, n_subpolicy_ops = 2):
        self.n_subpolicies = n_subpolicies
        self.n_subpolicy_ops = n_subpolicy_ops
        self.model = self.create_rnn_model()
        self.scale = tf.placeholder(tf.float32, ())
        self.grads = tf.gradients(self.model.outputs, self.model.trainable_weights)
        # negative value for gradient ascent
        self.grads = [g * (-self.scale) for g in self.grads]
        self.grads = zip(self.grads, self.model.trainable_weights)
        self.optimizer = tf.train.GradientDescentOptimizer(0.00035).apply_gradients(self.grads)

    def create_rnn_model(self):
        input_layer = layers.Input(shape=(self.n_subpolicies, 1))
        init = initializers.RandomUniform(-0.1, 0.1)
        lstm_layer = layers.LSTM(
            LSTM_UNITS, recurrent_initializer=init, return_sequences=True,
            name='controller')(input_layer)
        #lstm_layer = layers.Dense(50, activation='relu', name='middle_layer')(lstm_layer)
        outputs = []
        for i in range(self.n_subpolicy_ops):
            name = 'op%d-' % (i+1)
            outputs += [
                layers.Dense(OP_TYPES, activation='softmax', name=name + 't')(lstm_layer),
                layers.Dense(OP_PROBS, activation='softmax', name=name + 'p')(lstm_layer),
                layers.Dense(OP_MAGNITUDES, activation='softmax', name=name + 'm')(lstm_layer),
            ]
        return models.Model(input_layer, outputs)

    def fit(self, mem_softmaxes, mem_accuracies):
        sess = backend.get_session()
        saver = tf.train.Saver()
        min_acc = np.min(mem_accuracies)
        max_acc = np.max(mem_accuracies)
        dummy_input = np.zeros((1, self.n_subpolicies, 1))
        dict_input = {self.model.input: dummy_input}
        for softmaxes, acc in zip(mem_softmaxes, mem_accuracies):
            scale = (acc-min_acc) / (max_acc-min_acc)
            dict_outputs = {_output: s for _output, s in zip(self.model.outputs, softmaxes)}
            dict_scales = {self.scale: scale}
            sess.run(self.optimizer, feed_dict={**dict_outputs, **dict_scales, **dict_input})
        
        save_path = saver.save(sess, "./rl_models/model_rl_tt_2.ckpt")
        return self

    def predict(self, size):
        dummy_input = np.zeros((1, size, 1), np.float32)
        softmaxes = self.model.predict(dummy_input)
        # Convert softmaxes into subpolicies
        subpolicies = []
        for i in range(self.n_subpolicies):
            operations = []
            for j in range(self.n_subpolicy_ops):
                op = softmaxes[j*3:(j+1)*3]
                op = [o[0, i, :] for o in op]
                operations.append(Operation(*op))
            subpolicies.append(Subpolicy(*operations))
        return softmaxes, subpolicies
