import numpy as np
import pickle
import matplotlib.pyplot as plt
from models import Sequential
from layers import Dense
from layers import Sigmoid
from initializers import truncated_normal
from initializers import zeros
from metrics import SquaredError

image_size = 28
num_labels = 10
image_pixels = image_size**2

with open("pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

model = Sequential()
model.add(Dense(16, 784, kernel_initializer=truncated_normal, bias_initializer=zeros))
model.add(Sigmoid())
model.add(Dense(10, 16, kernel_initializer=truncated_normal, bias_initializer=zeros))
model.add(Sigmoid())

loss = SquaredError()

loss_history = model.fit(train_imgs, train_labels_one_hot, batch_size=32, epochs=10, loss=loss, halt=False)
pred = model.predict(test_imgs)
pred_labels = pred.argmax(1)
print("MSE", loss.evaluate(pred, test_labels_one_hot).mean(0))
print("Percentage correct", np.mean(pred_labels==test_labels)*100)
print("Prediction for first 5 images")
print(pred[0:5, :].argmax(1))
print("True labels")
print(test_labels[0:5])

plt.plot(np.arange(0, 10), loss_history.mean(1))
plt.title("Graph of mean loss over all one-hot outputs")
plt.xlabel("Epoch")
plt.ylabel("Mean loss")
plt.show()

print(model.save("mnist_model.pkl"))
