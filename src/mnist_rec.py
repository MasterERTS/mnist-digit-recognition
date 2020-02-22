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

output_layer = 10

mndata = MNIST('../res/')
mndata.gz = True
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

training_inputs = training_images.reshape(training_images.shape[0], training_images.shape[1] * training_images.shape[2]).astype('float32')
normalized_inputs = training_inputs/255
normalized_outputs = np.eye(output_layer)[training_labels]

testing_inputs = testing_images.reshape(testing_images.shape[0], testing_images.shape[1]*testing_images.shape[2]).astype('float32')
norm_test_inputs= testing_inputs/255
norm_test_outputs = testing_labels

layers = [784, 30, 10]
learning_rate = 0.05
batch_size = 1
epochs = 100