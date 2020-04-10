import numpy as np
from layers import Dense
from layers import Sigmoid
from models import Sequential
from initializers import saved_weights

kernel_weights = np.array([[15, -15],
                           [15, -15]])
bias_weights = np.array([-5, 5])

saved_kernel = saved_weights(kernel_weights)
saved_bias = saved_weights(bias_weights)

model = Sequential()
model.add(Dense(2, 2, kernel_initializer=saved_kernel, bias_initializer=saved_bias))
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

# [[ 6.27995387e-03 -6.27995387e-03]
#  [-4.83799975e-05  4.83799975e-05]
#  [-4.83799971e-05  4.83799971e-05]
#  [-1.47932777e-11  1.47933355e-11]]

print("Training")
print(model.fit(X, y, epochs=100, batch_size=2, steps_per_epoch=1000, halt=False))
print("Prediction")
p = model.predict(X)
print(p)
print("Error")
print(p-y)

# [[ 1.66552781e-04 -1.66552781e-04]
#  [-6.66199208e-05  6.66199208e-05]
#  [-6.66202493e-05  6.66202493e-05]
#  [-7.39408534e-13  7.39422160e-13]]

print("Weights")
print(model.layers[0].weights)

# [[ -8.70003173   8.70003173]
#  [ 18.31646709 -18.31646709]
#  [ 18.31647202 -18.31647202]]

# TODO: test with hidden layer
