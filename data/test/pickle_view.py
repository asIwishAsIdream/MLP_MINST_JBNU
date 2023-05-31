import sys, os
sys.path.append(os.pardir)
import pickle
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from common.layers import *

np.set_printoptions(linewidth=1000, threshold=100000)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

with open("MLP_Prams.pkl", 'rb') as f:
    MLP_network = pickle.load(f)

sample=x_test[5]
sample_label = t_test[5]
print(x_train.shape)
# Matplotlib 라이브러리를 사용하여 이미지 데이터를 시각화하는 코드
plt.figure()
plt.imshow(sample.reshape(28,28), cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()


probability = softmax(MLP_network.predict(sample.reshape(1,784)))
plt.bar(range(len(probability[0])), probability[0])
plt.ylim(0, 1.0)
plt.show()
print(sample)
print("Classification Accuracy : " + str(MLP_network.accuracy(sample.reshape(1,784),sample_label)))
