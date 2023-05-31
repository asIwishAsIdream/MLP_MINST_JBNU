import pickle

from dataset.mnist import load_mnist
from project1_ML.MLP import MLP
from common.functions import *


import matplotlib.pyplot as plt
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(t_test.shape)


sample=x_train[0]
plt.figure()
plt.imshow(sample.reshape(28,28), cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()


# 사용자가 은닉층의 차원수 + 은닉층의 개수를 설정할 수 있다

hidden_size_list = [100, 100, 100, 100]

net_MLP = MLP(input_size=784, hidden_size_list=hidden_size_list, output_size=10, use_dropout=True, use_batchnorm=True)

# mini bach 의 크기를 설정할 수 있다
iters_num = 1000
eval_interval = 50
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 1번의 에포크가 끝나는 시점 (에포크당 반복 시점)
iter_per_epoch = max(train_size / batch_size, 1)
# 사용자가 최대 epoch수를 설정할 수 있다
max_epoch_num = 10
max_epoch = int(iter_per_epoch * max_epoch_num)

# Early Stopping 관련 변수
min_delta = 0.001
patience = 2
best_val_loss = np.inf # 무한대를 나타내는 값, 최대 또는 최소 값을 찾을 때 초기 비교 대상으로 설정된다
couter = 0

for i in range(max_epoch):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]



    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = net_MLP.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # SGD 적용 + dropout, batch normalization 적용
    for j in range(1, net_MLP.all_size_list_num):
        net_MLP.params['W' + str(j)] -= learning_rate * grad['W' + str(j)]
        net_MLP.params['b' + str(j)] -= learning_rate * grad['b' + str(j)]

        if 'gamma' + str(j) in grad:
            net_MLP.params['gamma' + str(j)] -= learning_rate * grad['gamma' + str(j)]
            net_MLP.params['beta' + str(j)] -= learning_rate * grad['beta' + str(j)]


    train_accuracy = net_MLP.accuracy(x_batch, t_batch)

    # 1 epoch마다 accuracy 출력
    if i % iter_per_epoch == 0:
        print("Classification Accuracy : " + str(train_accuracy))

        print(f"Epoch {i // iter_per_epoch + 1}, Classification Accuracy: {train_accuracy}")

    if (i % eval_interval == 0) & ((i // eval_interval) < 16):
        probability = softmax(net_MLP.predict(sample.reshape(1, 784)))
        plt.subplot(4, 4, int((i // eval_interval) + 1))
        plt.bar(range(len(probability[0])), probability[0])
        plt.ylim(0, 1.0)

    # 검증 손실 계산
    val_loss = net_MLP.loss_function(x_batch, t_batch)
    print(f"Epoch {i // iter_per_epoch + 1}, Validation Loss: {val_loss}")

    # Early Stopping 적용
    if np.abs(val_loss - best_val_loss) > min_delta:
        best_val_loss = val_loss
        couter = 0
    else:
        couter += 1

    if couter >= patience:
        print(f"Early stopping at Epoch {i // iter_per_epoch + 1}, Best Validation Loss: {best_val_loss}")
        break

plt.show()




with open('data/test/MLP_Prams.pkl', 'wb') as f:
    pickle.dump(net_MLP, f)