import matplotlib.pyplot as plt

from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# print(x_train.shape) (60000, 784)

# print(t_test.shape) (10000, 10)

plt.imshow(x_test[120].reshape((28, 28)), cmap='gray')
plt.show()