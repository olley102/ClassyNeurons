import numpy as np
import dill
from metrics import CustomLoss


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
            loss = CustomLoss()  # default loss

        error_signal = loss.gradient(pred, y)

        # Backprop.
        for layer in self.layers[::-1]:
            error_signal = layer.backprop(error_signal)
            layer.update()

        return np.mean(loss.evaluate(self.predict(X), y), axis=0)

    def fit(self, X, y, loss=None, batch_size=32, epochs=1, steps_per_epoch=None, shuffle=True, halt=True):
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

            print("Mean losses:", loss_value)
            loss_history = np.append(loss_history, loss_value, axis=0)
            ep += 1

        return loss_history

    def save(self, file_path):
        data = []

        for layer in self.layers:
            data.append(layer.get_save_data())

        with open(file_path, "bw") as f:
            dill.dump(data, f)

        return "Model saved."

    def restore(self, file_path):
        with open(file_path, "br") as f:
            data = dill.load(f)

        self.layers = np.array([])

        for layer_data in data:
            assert isinstance(layer_data, dict)
            layer_cls = layer_data["class"]
            data_copy = layer_data.copy()
            data_copy.pop("class")
            self.layers = np.append(self.layers, layer_cls(**data_copy))

        return "Model restored."
