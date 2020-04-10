import numpy as np

class Dense:
    def __init__(self, output_dim:int, input_dim:int, kernel_initializer=None, bias_initializer=None, alpha=1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.zeros((input_dim+1, output_dim))
        self.initialize(kernel_initializer, bias_initializer)
        self.alpha = alpha
        self.Z = np.array([])
        self.neurons = np.array([])
        self.sigma = np.array([])
        self.delta = np.array([])

    def initialize(self, kernel_initializer=None, bias_initializer=None):
        if kernel_initializer is not None:
            self.weights[1:, :] = np.asmatrix(kernel_initializer(self.input_dim, self.output_dim))
        if bias_initializer is not None:
            self.weights[0, :] = np.asmatrix(bias_initializer(1, self.output_dim))

    def predict(self, X, bias_included=False):
        X_matrix = np.asmatrix(X)
        if bias_included:
            self.Z = X_matrix
        else:
            self.Z = np.ones((X_matrix.shape[0], X_matrix.shape[1]+1))
            self.Z[:, 1:] = X_matrix

        self.neurons = self.Z @ self.weights
        return self.neurons

    def backprop(self, sigma):
        self.sigma = np.asmatrix(sigma) @ self.weights.transpose()
        self.sigma = self.sigma[:, 1:]
        return self.sigma

    def weight_gradient(self, sigma):
        self.delta = self.Z.transpose() @ sigma
        return self.delta

    def update(self):
        self.weights = self.weights - (1.0/self.Z.shape[0]) * self.alpha * self.delta


class Activation:
    @staticmethod
    def activate(x):
        return x

    @staticmethod
    def derivative(x):
        return 1

    def __init__(self):
        self.Z = np.array([])
        self.neurons = np.array([])
        self.sigma = np.array([])

    def predict(self, X):
        self.Z = np.asmatrix(X)
        self.neurons = self.__class__.activate(X)
        return self.neurons

    def compute_gradient(self):
        self.sigma = self.__class__.derivative(self.Z)
        return self.sigma

    def backprop(self, sigma):
        return np.multiply(sigma, self.compute_gradient())


class Sigmoid(Activation):
    @staticmethod
    def activate(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return np.multiply(Sigmoid.activate(x), (1.0-Sigmoid.activate(x)))

    def __init__(self):
        super().__init__()


class ReLU(Activation):
    @staticmethod
    def activate(x):
        return x * (x > 0)

    @staticmethod
    def derivative(x):
        return (x > 0) * 1

    def __init__(self):
        super().__init__()
