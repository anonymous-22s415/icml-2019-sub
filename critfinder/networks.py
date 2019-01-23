from collections import namedtuple
import functools
import itertools
import json

import autograd
from autograd import numpy as np

XH_EPS = 1e-25


def relu(x):
    return np.where(x > 0., x, 0.)


def sigmoid(x):
    return np.where(x >= 0, _positive_sigm(x), _negative_sigm(x))


def _negative_sigm(x):
    expon = np.exp(-x)
    return 1 / (1 + expon)


def _positive_sigm(x):
    expon = np.exp(x)
    return expon / (1 + expon)


_NONLINEARITIES = {"relu": relu,
                   "sigmoid": sigmoid,
                   "none": lambda x: x}


def mean_squared_error(x, xhat):
    return np.mean(np.square(x - xhat))


def softmax_cross_entropy(l, p):
    phat = softmax(l)
    return np.mean(cross_entropy(p, phat))


def softmax(x):
    expon = np.exp(x - np.max(x, axis=1))
    return expon / np.sum(expon)


def cross_entropy(ps, qs, eps=XH_EPS):
    return -np.dot(ps.T, np.log(qs + eps))


_COSTS = {"mean_squared_error": mean_squared_error,
          "softmax_cross_entropy": softmax_cross_entropy}


def l2_regularizer(theta):
    return np.sum(np.square(theta))


def l1_regularizer(theta):
    return np.sum(np.abs(theta))


_REGULARIZERS = {"l2": l2_regularizer,
                 "l1": l1_regularizer,
                 "none": lambda x: 0.}


Data = namedtuple("Data", ['x', 'y'])


def pointwise_nonlinearity(parameters, x, nonlinearity):
    W, b = parameters
    return nonlinearity(np.dot(W, x) + b)


def gradient_descent(grad_f, init_theta, num_iters, lr=0.1):
    thetas = [init_theta]
    for ii in range(num_iters):
        theta_ii = thetas[ii]
        thetas.append(theta_ii - lr * grad_f(theta_ii))
    return thetas


def momentum(grad_f, init_theta, num_iters, lr=0.1, momentum=0.1):
    thetas = [init_theta]
    step = 0.
    for ii in range(num_iters):
        theta_ii = thetas[ii]
        step = -lr * grad_f(theta_ii) + momentum * step
        thetas.append(theta_ii + step)
    return thetas


class Network(object):

    def __init__(self, data, layers, cost_str="mean_squared_error", nonlinearity_str="relu",
                 regularizer_str="none", regularization_parameter=0., has_biases=True):
        if not isinstance(data, Data):
            try:
                data = Data(x=data[0], y=data[1])
            except IndexError:
                raise("data argument not understood")

        self.data = data

        self.cost_str = cost_str
        self.nonlinearity_str = nonlinearity_str
        self.regularizer_str = regularizer_str

        self.cost = _COSTS[self.cost_str]
        self.nonlinearity = _NONLINEARITIES[self.nonlinearity_str]
        self.regularizer = _REGULARIZERS[self.regularizer_str]

        self.regularization_parameter = regularization_parameter
        self.has_biases = has_biases
        self.layers = layers

        self.grad = autograd.grad(self.loss)
        self.hess = autograd.hessian(self.loss)

    def loss(self, theta):
        return (self.cost(self.output(self.data.x, theta), self.data.y) +
                self.regularization_parameter * self.regularizer(theta))

    def output(self, x, theta):
        raise NotImplementedError

    def to_json(self, filename):
        dictionary = self.construct_dict()
        with open(filename, "w") as f:
            json.dump(dictionary, f)

    @classmethod
    def from_json(cls, data, filename):
        with open(filename) as f:
            dictionary = json.load(f)
        return cls(data, **dictionary)

    def construct_dict(self):
        return {"layers": self.layers,
                "cost_str": self.cost_str,
                "nonlinearity_str": self.nonlinearity_str,
                "regularizer_str": self.regularizer_str,
                "regularization_parameter": self.regularization_parameter,
                "has_biases": self.has_biases}


class FeedforwardNetwork(Network):

    def __init__(self, data, layers, cost_str="mean_squared_error", nonlinearity_str="relu",
                 regularizer_str="none", regularization_parameter=0., has_biases=True):
        Network.__init__(self, data, layers, cost_str, nonlinearity_str, regularizer_str,
                         regularization_parameter, has_biases)
        for layer in layers:
            assert isinstance(layer, int)

    def output(self, x, theta):
        weights = self.extract_weights(theta)
        biases = self.extract_biases(theta)
        weights_and_biases = list(itertools.zip_longest(weights, biases,
                                                        fillvalue=0.))

        def network_nonlinearity(x, parameters):
            return pointwise_nonlinearity(parameters, x, self.nonlinearity)

        phi = functools.reduce(network_nonlinearity,
                               weights_and_biases[:-1],
                               x)

        out = pointwise_nonlinearity(weights_and_biases[-1], phi, lambda x: x)

        return out

    def extract_weights(self, theta):
        in_sizes = self.layers[:-1]
        out_sizes = self.layers[1:]
        counter = 0
        weight_matrices = []
        for in_size, out_size in zip(in_sizes, out_sizes):
            num_params = in_size * out_size
            weight_values = theta[counter:counter + num_params]
            weight_matrix = np.reshape(weight_values, (out_size, in_size))
            weight_matrices.append(weight_matrix)
            counter += num_params
        self.num_weights = counter
        return weight_matrices

    def extract_biases(self, theta):
        if not self.has_biases:
            return []
        else:
            bias_vectors = []
            counter = self.num_weights
            for layer_size in self.layers[1:]:
                num_params = layer_size
                bias_values = theta[counter:counter + num_params]
                bias_vector = np.atleast_2d(bias_values)
                bias_vectors.append(bias_vector)
                counter += num_params
            return bias_vectors
