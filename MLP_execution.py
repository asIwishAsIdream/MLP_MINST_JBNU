from dataset.mnist import load_mnist
from project1_ML.MLP import MLP
from common.functions import *

from torch.nn import functional as f
from torch.nn import Conv1d as tc

from sklearn import datasets


import pickle
import matplotlib.pyplot as plt
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arr')
print(t_test.shape)




##
sample=x_train[0]
# print(x_train.shape)
# Matplotlib 라이브러리를 사용하여 이미지 데이터를 시각화하는 코드
plt.figure()
plt.imshow(sample.reshape(28,28), cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()




# 사용자가 은닉층의 차원수 + 은닉층의 개수를 설정할 수 있다
hidden_size_list = [100, 100, 100, 100]

net_MLP = MLP(input_size=784, hidden_size_list=hidden_size_list, output_size=10)

# mini bach 의 크기를 설정할 수 있다
iters_num = 10000
eval_interval = 50
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 1번의 에포크가 끝나는 시점 (에포크당 반복 시점)
iter_per_epoch = max(train_size / batch_size, 1)
max_epoch = 100

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]



    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = net_MLP.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # SGD 적용
    for j in range(1, net_MLP.all_size_list_num):
        net_MLP.params['W' + str(j)] = learning_rate * grad['W' + str(j)]
        net_MLP.params['b' + str(j)] = learning_rate * grad['b' + str(j)]

    if i % iter_per_epoch == 0:
        print("Classification Accuracy : " + str(net_MLP.accuracy(x_batch,t_batch)))


    if (i % eval_interval == 0) & ((i // eval_interval) < 16):
        probability = softmax(net_MLP.predict(sample.reshape(1, 784)))
        print(probability)
        plt.subplot(4, 4, int((i // eval_interval) + 1))
        plt.bar(range(len(probability[0])), probability[0])
        plt.ylim(0, 1.0)
plt.show()

with open('MLP_Prams.pkl', 'wb') as f:
    pickle.dump(net_MLP, f)