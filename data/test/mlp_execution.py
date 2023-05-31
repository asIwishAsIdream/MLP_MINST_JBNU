# coding: utf-8
import sys, os

import matplotlib.pyplot as plt

sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from mlp_test import MLPTest
from common.functions import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# sample 데이터를 추출합니다
sample=x_test[5]
print(x_train.shape)
# Matplotlib 라이브러리를 사용하여 이미지 데이터를 시각화하는 코드
plt.figure()
plt.imshow(sample.reshape(28,28), cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()

hidden_size_list = [100, 100, 100, 100]

network = MLPTest(input_size=784, hidden_size_list=hidden_size_list, output_size=10)


iters_num = 1000
eval_interval = 50
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size / batch_size, 1)

plt.figure(figsize=(10,10))
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # 갱신
    for j in range(1, network.all_size_list_num):
        network.params['W' + str(j)] -= learning_rate * grad['W' + str(j)]
        network.params['b' + str(j)] -= learning_rate * grad['b' + str(j)]

    if i % iter_per_epoch == 0:
        print("Classification Accuracy : " + str(network.accuracy(x_batch,t_batch)))

    if (i % eval_interval == 0) & ((i//eval_interval)< 16):
        probability = softmax(network.predict(sample.reshape(1,784)))
        plt.subplot(4,4, int((i//eval_interval) + 1))
        plt.bar(range(len(probability[0])),probability[0])
        plt.ylim(0, 1.0)
plt.show()



