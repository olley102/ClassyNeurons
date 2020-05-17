import numpy as np
from abc import ABCMeta
from abc import abstractmethod
from initializers import saved_weights


class Layer(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, X):
        """
        Forward propagate.

        :param X: input data.
        :return: self.Z
        """
        pass

    @abstractmethod
    def backprop(self, error_signal):
        """
        Backward propagate.

        :param error_signal: signal from next layer.
        :return: self.error_signal_in
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update weights.
        """
        pass

    @abstractmethod
    def get_save_data(self):
        """
        Get data to store in binary file in models.

        :return: list of class, output_dim, input_dim, weights, ...
        """
        pass


class Neural(Layer, metaclass=ABCMeta):
    """
    Abstract class for layers with standard forward-prop and backprop.
    """
    def __init__(self, output_dim:int, input_dim:int):
        """
        Initialize layer.

        :param output_dim: number of output features.
        :param input_dim: number of input features.

        Attributes:
        - X: input data with size[0] equal to number of examples. Bias ones may be added.
        - Z: output data with size[0] equal to number of examples.
        - input_dim: number of input features.
        - output_dim: number of output features.
        - error_signal_in: gradient of loss w.r.t. X without bias.
        - error_signal_out: gradient of loss w.r.t. Z.
        - gradient: gradient of loss w.r.t. weights.
        """
        self.X = np.matrix([])
        self.Z = np.matrix([])
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.error_signal_in = np.matrix([])
        self.error_signal_out = np.matrix([])
        self.gradient = np.matrix([])

    @abstractmethod
    def predict(self, X):
        """
        Forward propagate.

        :param X: input data.
        :return: self.Z
        """
        pass

    @abstractmethod
    def backprop(self, error_signal):
        """
        Backward propagate.

        :param error_signal: signal from next layer.
        :return: self.error_signal_in
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update weights.
        """
        pass

    @abstractmethod
    def get_save_data(self):
        """
        Get data to store in binary file in models.

        :return: list of class, output_dim, input_dim, weights, ...
        """
        pass


class Dense(Neural):
    def __init__(self, output_dim:int, input_dim:int, kernel_initializer=None, bias_initializer=None, alpha=1.0):
        """
        Initialize Dense layer.

        :param output_dim: number of output features.
        :param input_dim: number of input features.
        :param kernel_initializer: function to call to initialize non-bias weights.
        :param bias_initializer: function to call to initialize bias weights.
        :param alpha: learning rate.
        """
        super().__init__(output_dim, input_dim)
        self.weights = np.asmatrix(np.zeros((input_dim+1, output_dim)))
        self.initialize(kernel_initializer, bias_initializer)
        self.alpha = alpha

    def initialize(self, kernel_initializer=None, bias_initializer=None):
        """
        Initialize weights.

        :param kernel_initializer: function to call to initialize non-bias weights.
        :param bias_initializer: function to call to initialize bias weights.
        """
        if kernel_initializer is not None:
            self.weights[1:, :] = np.asmatrix(kernel_initializer(self.input_dim, self.output_dim))
        if bias_initializer is not None:
            self.weights[0, :] = np.asmatrix(bias_initializer(1, self.output_dim))

    def predict(self, X, bias_included=False):
        """
        Forward propagate.

        :param X: input data.
        :param bias_included: bias ones already in X.
        :return: self.Z
        """
        X_matrix = np.asmatrix(X)
        if bias_included:
            self.X = X_matrix
        else:
            self.X = np.ones((X_matrix.shape[0], X_matrix.shape[1] + 1))
            self.X[:, 1:] = X_matrix

        self.Z = self.X @ self.weights
        return self.Z

    def backprop(self, error_signal):
        """
        Backward propagate.

        :param error_signal: signal from next layer.
        :return: self.error_signal_in
        """
        self.error_signal_out = np.asmatrix(error_signal)
        self.error_signal_in = self.error_signal_out @ self.weights.transpose()
        self.error_signal_in = self.error_signal_in[:, 1:]
        return self.error_signal_in

    def update(self):
        """
        Update weights.
        """
        self.gradient = self.X.transpose() @ self.error_signal_out
        self.weights = self.weights - self.alpha * (1/self.X.shape[0]) * self.gradient

    def get_save_data(self):
        """
        Get data to store in binary file in models.

        :return: list of class, output_dim, input_dim, weights, ...
        """
        data = {
            "class": self.__class__,
            "output_dim": self.output_dim,
            "input_dim": self.input_dim,
            "kernel_initializer": saved_weights(self.weights[1:, :]),
            "bias_initializer": saved_weights(self.weights[0, :])
        }

        return data


class Activation(Layer, metaclass=ABCMeta):
    """
    Abstract class for activation layers.
    """
    def __init__(self):
        self.X = np.matrix([])
        self.Z = np.matrix([])
        self.error_signal_in = np.matrix([])
        self.error_signal_out = np.matrix([])

    @abstractmethod
    def activate(self, x):
        """
        Activation function goes here.

        :param x: matrix to activate.
        :return: activation of x.
        """
        pass

    @abstractmethod
    def derivative(self, x):
        """
        Activation derivative goes here.

        :param x: matrix argument to derivative function.
        :return: derivative of activation at x.
        """
        pass

    def update(self):
        """
        Update does nothing. No weights to update.
        """
        pass

    def predict(self, X):
        self.X = np.asmatrix(X)
        self.Z = self.activate(X)
        return self.Z

    def backprop(self, error_signal):
        self.error_signal_out = error_signal
        self.error_signal_in = np.multiply(error_signal, self.derivative(self.X))
        return self.error_signal_in

    def get_save_data(self):
        """
        Get data to store in binary file in models.

        :return: list of class, output_dim, input_dim, weights, ...
        """
        data = {
            "class": self.__class__
        }

        return data


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
