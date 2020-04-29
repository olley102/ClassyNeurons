import numpy as np

class Loss:
    def __init__(self, loss=None, derivative=None, dp=0.1, centering=0):
        if loss is not None:
            self.loss = loss
            if derivative is not None:
                self.derivative = derivative
            else:
                # Symmetric derivative
                if centering == 0:
                    self.derivative = lambda p : (loss(p+0.5*dp) - loss(p-0.5*dp)) / dp
                # Right derivative
                elif centering == 1:
                    self.derivative = lambda p : (loss(p+dp) - loss(p)) / dp
                # Left derivative
                elif centering == -1:
                    self.derivative = lambda p : (loss(p) - loss(p-dp)) / dp
                else:
                    raise ValueError("Centering value is invalid. Must be integer of 0, 1 or -1.")
        else:
            self.loss = lambda p, y, x : (1/(2*y.shape[0])) * ((p-y).transpose() @ (p-y)).diagonal()
            self.derivative = lambda p, y, x : (1/y.shape[0]) * (x.transpose() @ (p-y)).sum(0)

    def evaluate(self, pred, y, X):
        return self.loss(np.asmatrix(pred), np.asmatrix(y), np.asmatrix(X))

    def gradient(self, pred, y, X):
        # print("Shape", (pred-y).shape)
        return self.derivative(np.asmatrix(pred), np.asmatrix(y), np.asmatrix(X))


class MeanSquaredError(Loss):
    def __init__(self):
        loss = lambda p, y, x : (1/(2*y.shape[0])) * ((p-y).transpose() @ (p-y)).diagonal()
        derivative = lambda p, y, x : (1/y.shape[0]) * (x.transpose() @ (p-y)).sum(0)
        super().__init__(loss=loss, derivative=derivative)


class MeanAbsoluteError(Loss):
    def __init__(self):
        loss = lambda p, y, x : (1/len(y)) * np.abs(p-y).sum(0)
        derivative = lambda p, y, x : (1/y.shape[0]) * (x.transpose() @ (np.multiply(p-y, (1/np.abs(p-y))))).sum(0)
        super().__init__(loss=loss, derivative=derivative)
