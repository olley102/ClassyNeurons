import numpy as np
import pickle
from models import Sequential
from metrics import SquaredError
import matplotlib.pyplot as plt

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
model.restore("mnist_model.pkl")

loss = SquaredError()

pred = model.predict(test_imgs)
pred_labels = pred.argmax(1)
print("MSE", loss.evaluate(pred, test_labels_one_hot).mean(0))
print("Percentage correct", np.mean(pred_labels==test_labels)*100)
print("Prediction for first 5 images")
print(pred[0:5, :].argmax(1))
print("True labels")
print(test_labels[0:5])

fig, ax = plt.subplots(2, 5)

for i, ax in enumerate(ax.flatten()):
    im_idx = np.argwhere(test_labels == i)[0, 0]
    print(test_imgs[im_idx, :].shape)
    img = np.reshape(test_imgs[im_idx, :], (28, 28))
    ax.imshow(img, cmap="gray_r")
    print("Prediction:", model.predict(test_imgs[im_idx, :]).argmax(1)[0,0])
    print("True:", i)

plt.show()
