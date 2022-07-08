import matplotlib
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_train[0])
print(y_train[0])#5

import matplotlib.pyplot as plt
plt.imshow(x_train[5], 'gray')
plt.show()