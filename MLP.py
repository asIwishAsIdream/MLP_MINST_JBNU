# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import logging

from collections import OrderedDict
from common.layers import *



class MLP:

    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_decay_lambda=0, weight_init_std=0.01,
                 use_dropout=False, dropout_ration=0.5, use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.all_size_list_num = None
        self.params = {}  # 학습을 해야할 것들 Weights 와 biases
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)
        self. weight_init_std = weight_init_std
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.dropout_ration = dropout_ration


        # 가중치 초기화
        self.__init_weight()


        # 계층 생성
        activation_layer = {'relu': Relu}
        self.layers = OrderedDict()
        # Affine층 뿐만 아니라 활성화 층도 생성을 시켜야하기 때문에,
        for idx in range(1, self.hidden_layer_num + 1):
            # 사용자가 지정한 layer의 크기만큼 layers가 생성된다
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])

            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx - 1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx - 1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)],
                                                                         self.params['beta' + str(idx)])

            # Relu 클래스의 인스턴스가 layers에 담기게 된다
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        # 마지막 -1 층까지 만들어준다
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])
        # last_layer는 따로 설정하지 않고 gradient에서 미분으로 진행한다
        # self.last_layer = SoftmaxWithLoss()

    def __init_weight(self):
        # 리스트이므로 벡터의 덧셈이 아니라 [784, 100, 100,100, 10] 이라는 리스트가 생성됨
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        self.all_size_list_num = len(all_size_list)
        for idx in range(1, len(all_size_list)):  # all_size_list 는 5개의 길이를 가진 리스트이다
            scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLU를 사용할 때의 권장 초깃값

            # 표준정규 분포를 따라서 행렬을 만든다 처음은 784x100 행렬을 만들고 scale을 곱해줘라
            # 그러면 N(0, scale) 값으로 바뀌게 된다 (N은 정규분포를 의미)
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.random.randn(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            x = layer.forward(x)

        return x

    def loss_function(self, x, t, train_flg=False):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x, train_flg)

        # L2 패널티를 나타낸다
        weight_decay = 0
        # 3개의 층이므로 1 ~ 4 까지 반복한다
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            # <L2 패널티>, 람다식에 대한 총합을 나타낸다
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        # softmax_forward는 not-likelyhood의 값을 나타냄 + L2 패널티 값 == not likelyhood결과
        return self.softmax_forward(y, t) + weight_decay

    def softmax_forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = negative_log_likelihood_loss(self.y, self.t)
        # self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def loss_for_backward(self):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # Backward에 해당
    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 딕셔너리(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss_function(x, t, train_flg=True)


        # backward (맨 마지막은 흘러 들어오는 게 없음)
        # dout = 1
        # 크로스 엔트로피만 고려해서 역전파로 기울기를 넘겨주는 것임
        # 여기서 loss까지 같이 계산되므로 위해서 loss_backward를 할 필요가 없다
        dout = self.loss_for_backward()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)


        # Result
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params[
                'W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta


        return grads

