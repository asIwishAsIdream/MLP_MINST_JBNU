# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
# 저장을 위함
import pickle

class MLPTest:

    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_decay_lambda=0, weight_init_std=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.all_size_list_num = None
        self.params = {}  # 학습을 해야할 것들 Weights 와 biases
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)
        self.weight_init_std = weight_init_std

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
            # Relu 클래스의 인스턴스가 layers에 담기게 된다
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
        # 마지막 -1 층까지 만들어준다
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])
        # last_layer는 따로 설정하지 않고 gradient에서 미분으로 진행한다
        self.lastlayer = SoftmaxWithLoss()

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
        return self.lastlayer.forward(y, t)

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
        dout = self.lastlayer.backward(dout)

        # layers에 있는 각 layer에 역전파를 해줘야함
        # Affine1, ReLU, Affine2 가 layers에 담김
        layers = list(self.layers.values())
        # 역전파를 위해서 layer들을 뒤집어 준다
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
        # grads로 SGD를 가능하게 만든다
        return grads

