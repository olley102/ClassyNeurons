import numpy as np
from layers import Dense
from layers import Sigmoid
from models import Sequential
from initializers import saved_weights
from metrics import MeanSquaredError
import matplotlib.pyplot as plt

kernel_weights = np.array([[0, 0],
                           [0, 0]])
bias_weights = np.array([0, 0])

saved_kernel = saved_weights(kernel_weights)
saved_bias = saved_weights(bias_weights)

model = Sequential()
model.add(Dense(2, 2, kernel_initializer=saved_kernel, bias_initializer=saved_bias, alpha=2.5))
model.add(Sigmoid())

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0, 1],
              [1, 0],
              [1, 0],
              [1, 0]])

print("Prediction")
p = model.predict(X)
print(p)
print("Error")
print(p-y)

loss_function = MeanSquaredError()

print("Training")
loss_history = model.fit(X, y, epochs=100, batch_size=2, steps_per_epoch=1000, halt=False, loss=loss_function)
print("Prediction")
p = model.predict(X)
print(p)
print("Error")
print(p-y)
print("Weights")
print(model.layers[0].weights)

plt.plot(np.arange(0, 100), loss_history[:, 0])
plt.plot(np.arange(0, 100), loss_history[:, 1])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
