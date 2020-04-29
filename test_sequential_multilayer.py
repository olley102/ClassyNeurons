import numpy as np
from layers import Dense
from layers import Sigmoid
from models import Sequential
from initializers import saved_weights
from metrics import MeanSquaredError
import matplotlib.pyplot as plt

bias_weights_0 = np.array([-5.1, 5])
kernel_weights_0 = np.array([[5, -5.1],
                             [5, -5.1]])

bias_weights_1 = np.array([-5.1])
kernel_weights_1 = np.array([[5.1],
                             [5]])

saved_bias_0 = saved_weights(bias_weights_0)
saved_kernel_0 = saved_weights(kernel_weights_0)

saved_bias_1 = saved_weights(bias_weights_1)
saved_kernel_1 = saved_weights(kernel_weights_1)

model = Sequential()
model.add(Dense(2, 2, kernel_initializer=saved_kernel_0, bias_initializer=saved_bias_0, alpha=1.25))
model.add(Sigmoid())
model.add(Dense(1, 2, kernel_initializer=saved_kernel_1, bias_initializer=saved_bias_1, alpha=1.25))
model.add(Sigmoid())

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[1],
              [0],
              [0],
              [1]])

print("Prediction")
p = model.predict(X)
print(p)
print("Error")
print(p-y)

loss_function = MeanSquaredError()

print("Training")
loss_history = model.fit(X, y, epochs=100, batch_size=4, steps_per_epoch=1000, halt=False, loss=loss_function)
print("Prediction")
p = model.predict(X)
print(p)
print("Error")
print(p-y)
print("Weights in first dense layer")
print(model.layers[0].weights)
print("Weights in second dense layer")
print(model.layers[2].weights)

plt.plot(np.arange(0, 100), loss_history[:, 0])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
