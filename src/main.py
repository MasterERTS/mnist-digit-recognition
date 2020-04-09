import numpy as np
import math
import random
import matplotlib.pyplot as plt
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import mnist_parser as mnist
from lib.neural_network import NeuralNetwork

def print_random_image():
    testing_images = mnist.test_images()
    testing_labels = mnist.test_labels()

    picked_image = random.randint(0, min(len(testing_images), len(testing_labels)))

    img = testing_images[picked_image].reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.title(testing_labels[picked_image])
    plt.show()

def plot_loss(NeuralNetwork):
    y = NeuralNetwork.loss
    plt.plot(y)
    plt.ylabel('Loss function')
    plt.xlabel('Number of iterations')
    plt.xticks(rotation=60)
    plt.title('Loss function w.r.t. number of iterations')
    plt.show()

def plot_accuracy(NeuralNetwork):
    y = NeuralNetwork.accuracy
    plt.plot(y)
    plt.ylabel('Accuracy function')
    plt.xlabel('Number of iterations')
    plt.xticks(rotation=60)
    plt.title('Accuracy function w.r.t. number of iterations')
    plt.show()

def main():
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
    learning_rate = 0.001
    batch_size = 1
    epochs = 5

    nn = NeuralNetwork(layers, batch_size, epochs, learning_rate)
    nn.fit(normalized_inputs, normalized_outputs)
    nn.time_stamps_test()

if __name__ == "__main__":
    main()
