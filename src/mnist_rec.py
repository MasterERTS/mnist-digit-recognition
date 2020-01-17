import numpy as np
import math
import random
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from lib.util import *

from mnist import MNIST

# https://github.com/kdexd/digit-classifier/blob/master/main.py
# https://stackoverflow.com/questions/44115411/mnist-training-error-in-for-loop
# https://github.com/louisjc/mnist-neural-network

mndata = MNIST('/home/erwin/Documents/coro/artin-neuralnetworks/data')
mndata.gz = True
images, labels = mndata.load_training()

train_inputs = [np.reshape(x, (784, 1)) for x in images[0]]
train_results = [vectorized_result(y) for y in images[1]]
val_inputs = [np.reshape(x, (784, 1)) for x in labels[0]]
val_results = labels[1]

layers = [784, 30, 10]
learning_rate = 0.05

w1 = np.random.rand(784, 1)
w2 = 0

a0 = images[0]
a1 = sigmoid(w1*a0)