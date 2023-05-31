import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist

np.set_printoptions(linewidth=1000, threshold=100000)

with open("neuralnet.pkl", 'rb') as f:
    network = pickle.load(f)

W1, W2 = network['W1'], network['W2']
b1, b2 = network['b1'], network['b2']

print(W1)

print(type(network))
print(network.keys())
print('W1 shape' + str(W1.shape))
print('W2 shape' + str(W2.shape))
print('b1 shape' + str(b1.shape))
print('b2 shape' + str(b2.shape))
