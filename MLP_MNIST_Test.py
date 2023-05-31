import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import *
import matplotlib.pyplot as plt

# 긴 배열이나 행렬을 읽기 쉽게 출력할 수 있습니다.
np.set_printoptions(linewidth=1000, threshold=100000)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

with open('data/test/MLP_Prams.pkl', 'rb') as f:
    net_MLP = pickle.load(f)

sample=x_test[4]
print(x_train.shape)
# Matplotlib 라이브러리를 사용하여 이미지 데이터를 시각화하는 코드
plt.figure()
plt.imshow(sample.reshape(28,28), cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()

# mini bach 의 크기를 설정할 수 있다
iters_num = 1000
eval_interval = 50
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    if i % iter_per_epoch == 0:
        print("Classification Accuracy : " + str(net_MLP.accuracy(x_batch,t_batch)))

    if (i % eval_interval == 0) & ((i // eval_interval) < 16):
        probability = softmax(net_MLP.predict(sample.reshape(1, 784)))
        print(probability)
        plt.subplot(4, 4, int((i // eval_interval) + 1))
        plt.bar(range(len(probability[0])), probability[0])
        plt.ylim(0, 1.0)
plt.show()