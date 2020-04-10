import numpy as np

class Sequential:
    def __init__(self):
        self.layers = np.array([])

    def add(self, layer):
        self.layers = np.append(self.layers, layer)

    def predict(self, X):
        pred = X

        for layer in self.layers:
            pred = layer.predict(pred)

        return pred

    def train_on_batch(self, X, y):
        pred = self.predict(X)
        sigma = pred - y  # assuming C is non-regularized mean squares

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

    def fit(self, X, y, batch_size=32, epochs=1, steps_per_epoch=None, shuffle=True, halt=True):
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

        while ep < epochs:
            if halt:
                response = input("Press enter to continue.")
            print("Running epoch", ep)

            if shuffle:
                order = np.arange(0, X.shape[0])
                np.random.shuffle(order)
                X_copy = X[order, :]
                y_copy = y[order, :]

            for step in range(steps_per_epoch):
                start = step * batch_size % X.shape[0]
                end = start + batch_size
                self.train_on_batch(X_copy[start:end, :], y_copy[start:end])

            ep += 1
