import numpy as np
from abc import ABCMeta
from abc import abstractmethod


class Neural(metaclass=ABCMeta):
    def __init__(self, output_dim, input_dim):
        self.X = np.matrix([])
        self.Z = np.matrix([])
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.error_signal = np.matrix([])
        self.delta = np.matrix([])

    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def backprop(self, error_signal):
        pass

    @abstractmethod
    def gradient(self, error_signal):
        pass

    @abstractmethod
    def update(self):
        pass


class Dense(Neural):
    def __init__(self, output_dim:int, input_dim:int, kernel_initializer=None, bias_initializer=None, alpha=1.0):
        super().__init__(output_dim, input_dim)
        self.weights = np.asmatrix(np.zeros((input_dim+1, output_dim)))
        self.initialize(kernel_initializer, bias_initializer)
        self.alpha = alpha

    def initialize(self, kernel_initializer=None, bias_initializer=None):
        if kernel_initializer is not None:
            self.weights[1:, :] = np.asmatrix(kernel_initializer(self.input_dim, self.output_dim))
        if bias_initializer is not None:
            self.weights[0, :] = np.asmatrix(bias_initializer(1, self.output_dim))

    def predict(self, X, bias_included=False):
        X_matrix = np.asmatrix(X)
        if bias_included:
            self.X = X_matrix
        else:
            self.X = np.ones((X_matrix.shape[0], X_matrix.shape[1] + 1))
            self.X[:, 1:] = X_matrix

        self.Z = self.X @ self.weights
        return self.Z

    def backprop(self, error_signal):
        self.error_signal = np.asmatrix(error_signal) @ self.weights.transpose()
        self.error_signal = self.error_signal[:, 1:]
        return self.error_signal

    def gradient(self, error_signal):
        self.delta = self.X.transpose() @ error_signal
        return self.delta

    def update(self):
        self.weights = self.weights - self.alpha * self.delta


class Activation(metaclass=ABCMeta):
    def __init__(self):
        self.X = np.matrix([])
        self.Z = np.matrix([])
        self.error_signal = np.matrix([])

    @abstractmethod
    def activate(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

    def predict(self, X):
        self.X = np.asmatrix(X)
        self.Z = self.activate(X)
        return self.Z

    def gradient(self):
        self.error_signal = self.derivative(self.X)
        return self.error_signal

    def backprop(self, error_signal):
        return np.multiply(error_signal, self.gradient())


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def activate(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        return np.multiply(self.activate(x), (1.0 - self.activate(x)))


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def activate(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return (x > 0) * 1
