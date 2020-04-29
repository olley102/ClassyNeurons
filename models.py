import numpy as np
from metrics import Loss

class Sequential:
    def __init__(self):
        self.layers = np.array([])

    def add(self, layer):
        self.layers = np.append(self.layers, layer)

    def predict(self, X):
        pred = np.asmatrix(X)

        for layer in self.layers:
            pred = layer.predict(pred)

        return pred

    def train_on_batch(self, X, y, loss=None):
        pred = self.predict(X)

        if loss is None:
            loss = Loss()

        sigma = loss.gradient(pred, y, X)

        for layer in self.layers[::-1]:
            sigma = layer.backprop(sigma)

        for level in range(len(self.layers)):
            if self.layers[level].__class__.__name__ == 'Dense':
                for next_layer in self.layers[level+1:]:
                    if next_layer.__class__.__name__ == 'Dense':
                        delta = self.layers[level].weight_gradient(next_layer.sigma)
                        break
                else:
                    delta = self.layers[level].weight_gradient(pred-y)

                self.layers[level].update()

        return loss.evaluate(self.predict(X), y, X)

    def fit(self, X, y, batch_size=32, epochs=1, steps_per_epoch=None, shuffle=True, halt=True, loss=None):
        if batch_size > X.shape[0]:
            batch_size = 1

        if X.shape[0] % batch_size == 0:
            if steps_per_epoch is None:
                steps_per_epoch = int(X.shape[0] / batch_size)
        else:
            steps_per_epoch = 1

        X_copy = X
        y_copy = y

        ep = 0
        loss_history = np.empty((0, y.shape[1]))

        while ep < epochs:
            if halt:
                response = input("Press enter to continue.")

            print("Running epoch", ep)

            if shuffle:
                order = np.arange(0, X.shape[0])
                np.random.shuffle(order)
                X_copy = X[order, :]
                y_copy = y[order, :]

            loss_value = 0

            for step in range(steps_per_epoch):
                start = step * batch_size % X.shape[0]
                end = start + batch_size
                loss_value = self.train_on_batch(X_copy[start:end, :], y_copy[start:end], loss=loss)

            print("Losses:", loss_value)
            loss_history = np.append(loss_history, loss_value, axis=0)
            ep += 1

        return loss_history
