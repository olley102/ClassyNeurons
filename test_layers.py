import numpy as np
from layers import Dense
from layers import Sigmoid
from initializers import saved_weights

kernel_weights = np.array([[20, -20],
                           [20, -20]])
bias_weights = np.array([-10, 10])

saved_kernel = saved_weights(kernel_weights)
saved_bias = saved_weights(bias_weights)

layer = Dense(2, 2, kernel_initializer=saved_kernel, bias_initializer=saved_bias)
sigmoid = Sigmoid()

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0, 1],
              [1, 0],
              [1, 0],
              [1, 0]])

print("Round 1")
z = layer.predict(X)
print("Z", z)
a = sigmoid.predict(z)
print("a", a)

error_signal = a - y
delta = layer.backprop(error_signal)
print("delta", delta)

layer.update()

print("Round 2")
z = layer.predict(X)
print("z", z)
a = sigmoid.predict(z)
print("a", a)

error_signal = a - y
delta = layer.backprop(error_signal)
print("delta", delta)

print("theta", layer.weights)
