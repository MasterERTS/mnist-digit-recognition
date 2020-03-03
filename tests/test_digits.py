import numpy as np
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import mnist_parser as mnist
from lib.neural_network import NeuralNetwork

def test_ones():
    output_layer = 10

    training_images = mnist.train_images()
    training_labels = mnist.train_labels()
    testing_images = mnist.test_images()
    testing_labels = mnist.test_labels()

    training_inputs = training_images.reshape(
        training_images.shape[0], training_images.shape[1] * training_images.shape[2]).astype('float32')
    normalized_inputs = training_inputs/255
    normalized_outputs = np.eye(output_layer)[training_labels]

    testing_inputs = testing_images.reshape(
        testing_images.shape[0], testing_images.shape[1]*testing_images.shape[2]).astype('float32')
    norm_test_inputs = testing_inputs/255
    norm_test_outputs = testing_labels

    layers = [784, 30, 10]

    learning_rate = 0.01
    batch_size = 1
    epochs = 5

    nn = NeuralNetwork(layers, batch_size, epochs, learning_rate)
    nn.fit(normalized_inputs, normalized_outputs, False)

    ones_filter = np.where(norm_test_outputs == 1)
    ones_test_labels = norm_test_outputs[ones_filter]
    ones_test_digit = norm_test_inputs[ones_filter]

    acc = nn.accuracy_test(ones_test_digit, ones_test_labels)
    assert(acc > 80)