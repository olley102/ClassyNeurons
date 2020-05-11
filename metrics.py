import numpy as np
from abc import ABCMeta
from abc import abstractmethod


class Loss(metaclass=ABCMeta):
    @abstractmethod
    def loss(self, p, y):
        pass

    @abstractmethod
    def derivative(self, p, y):
        pass

    def evaluate(self, pred, y):
        return self.loss(np.asmatrix(pred), np.asmatrix(y))

    def gradient(self, pred, y):
        return self.derivative(np.asmatrix(pred), np.asmatrix(y))


class CustomLoss(Loss):
    def __init__(self, loss=None, derivative=None, dp=0.1, centering=0):
        if loss is not None:
            self.custom_loss = loss
            if derivative is not None:
                self.custom_derivative = derivative
            else:
                # Symmetric derivative
                if centering == 0:
                    self.custom_derivative = \
                        lambda p, y : (self.custom_loss(p+0.5*dp, y) - self.custom_loss(p-0.5*dp, y)) / dp
                # Right derivative
                elif centering == 1:
                    self.custom_derivative = lambda p, y : (self.custom_loss(p+dp, y) - self.custom_loss(p, y)) / dp
                # Left derivative
                elif centering == -1:
                    self.custom_derivative = lambda p, y : (self.custom_loss(p, y) - self.custom_loss(p-dp, y)) / dp
                else:
                    raise ValueError("Centering value is invalid. Must be integer of 0, 1 or -1.")
        else:
            self.custom_loss = SquaredError().loss
            self.custom_derivative = SquaredError().derivative

    def loss(self, p, y):
        return self.custom_loss(p, y)

    def derivative(self, p, y):
        return self.custom_derivative(p, y)


class SquaredError(Loss):
    def loss(self, p, y):
        return 0.5 * np.square(p-y)

    def derivative(self, p, y):
        return p - y


class AbsoluteError(Loss):
    def loss(self, p, y):
        return np.abs(p-y)

    def derivative(self, p, y):
        return np.multiply(p-y, 1/np.abs(p-y))
