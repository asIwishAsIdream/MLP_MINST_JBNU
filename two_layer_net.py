# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
# 저장을 위함
import pickle

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {} # 초기 위치를 딕셔너리에 저장함
        # weight_init_std를 통해서 N(0, 0.01) 평균은 0이고 표준편차는 0.01을 따르는 걸 만들겠다
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict() # 순서가 있는 딕셔너리 사용 --> 딕셔너리는 순서에 구애받지 않음 그래서 a==b를 비교하면 True 로 같은 걸로 뜸
        # Affine 1 : Affine 클래스의 인스턴스
        # ReLU 1 : ReLU 클래스의 인스턴스
        # Affine 2 : Affine 클래스의 인스턴스 2
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    # softmax 전까지 하는 게 예측임
    def predict(self, x):
        # 각 layer에는 Affine1, Relu1, Affine2 가 대입됨
        for layer in self.layers.values():
            # 각 layers에 Affine1, Relu, Affine2를 지나 생성된 x의 값
            x = layer.forward(x)

        return x

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1) # 제일 큰 값을 y 에 저장시키겠다
        # t.ndim - dimention이 뭔지, 1이 아니면 - one hot encoding이 되있는 거임
        # 그래서 원핫 인코딩이 안되 있는 걸로 바꿔달라는 코드
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    # 미분의 정의
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    # 공식으로 미분
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        # 맨 마지막 층에서 들어온 미분값, 다음 층이  없으므로 1이 들어온다고 가정,
        dout = 1
        # lastLayer == softmaxWithLoss()의 인스턴스
        # 여기에 역전파가 있다 y - t : (확률분포 - 라벨값)을 밑으로 흘려보내는 것을 의미
        dout = self.lastLayer.backward(dout)

        # layers에 있는 각 layer에 역전파를 해줘야함
        # Affine1, ReLU, Affine2 가 layers에 담김
        layers = list(self.layers.values())
        # 역전파를 위해서 layer들을 뒤집어 준다
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        # grads로 SGD를 가능하게 만든다
        return grads

