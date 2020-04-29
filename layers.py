import numpy as np

class Dense:
    def __init__(self, output_dim:int, input_dim:int, kernel_initializer=None, bias_initializer=None, alpha=1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.zeros((input_dim+1, output_dim))
        self.initialize(kernel_initializer, bias_initializer)
        self.alpha = alpha
        self.X = np.array([])
        self.Z = np.array([])
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
            self.X = X_matrix
        else:
            self.X = np.ones((X_matrix.shape[0], X_matrix.shape[1] + 1))
            self.X[:, 1:] = X_matrix

        self.Z = self.X @ self.weights
        return self.Z

    def backprop(self, sigma):
        self.sigma = np.asmatrix(sigma) @ self.weights.transpose()
        self.sigma = self.sigma[:, 1:]
        return self.sigma

    def weight_gradient(self, sigma):
        self.delta = self.X.transpose() @ sigma
        return self.delta

    def update(self):
        self.weights = self.weights - (1.0 / self.X.shape[0]) * self.alpha * self.delta


class Activation:
    def __init__(self, activate=None, derivative=None):
        self.X = np.array([])
        self.Z = np.array([])
        self.sigma = np.array([])

        if activate is None:
            self.activate = lambda x : x
        else:
            self.activate = activate

        if derivative is None:
            self.derivative = lambda x : 1
        else:
            self.derivative = derivative

    def predict(self, X):
        self.X = np.asmatrix(X)
        self.Z = self.activate(X)
        return self.Z

    def compute_gradient(self):
        self.sigma = self.derivative(self.X)
        return self.sigma

    def backprop(self, sigma):
        return np.multiply(sigma, self.compute_gradient())


class Sigmoid(Activation):
    def __init__(self):
        activate = lambda x : 1.0 / (1.0 + np.exp(-x))
        derivative = lambda x : np.multiply(activate(x), (1.0-activate(x)))
        super().__init__(activate=activate, derivative=derivative)


class ReLU(Activation):
    def __init__(self):
        activate = lambda x : x * (x > 0)
        derivative = lambda x : (x > 0) * 1
        super().__init__(activate=activate, derivative=derivative)
