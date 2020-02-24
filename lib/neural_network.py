#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import lib.maths_util as mathlib
from lib.colors import ColorsBook as color
import time


class NeuralNetwork():
    def __init__(self, layers, batch_size, epochs, learning_rate):
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.weights = []
        self.biases = []
        self.loss = []
        self.accuracy = []
        n_layers = len(layers)

        for i in range(n_layers - 1):
            self.weights.append(np.random.normal(
                0, 1, [self.layers[i], self.layers[i+1]]))
            self.biases.append(np.zeros((1, self.layers[i+1])))

    def feed_forward(self, inputs):

        layer0 = inputs
        layer1 = mathlib.relu(np.dot(layer0, self.weights[0]) + self.biases[0])
        layer2 = mathlib.softmax(
            np.dot(layer1, self.weights[1]) + self.biases[1])

        return layer1, layer2

    def loss_function(self, predicted_outputs, outputs):
        loss = mathlib.cross_entropy(predicted_outputs, outputs)
        loss += mathlib.regularization(0.01, self.weights[0], self.weights[1])

        return loss

    def accuracy_function(self, predicted_outputs, outputs):
        acc = float(np.sum(np.argmax(predicted_outputs, 1) == outputs)) / \
            float(len(outputs))
        return acc

    def back_propagate(self, inputs, hidden_layer, predicted_outputs, outputs):
        delta_y = (predicted_outputs - outputs) / predicted_outputs.shape[0]
        delta_hidden_layer = np.dot(delta_y, self.weights[1].T)
        delta_hidden_layer[hidden_layer <= 0] = 0

        w2_grad = np.dot(hidden_layer.T, delta_y)
        b2_grad = np.sum(delta_y, axis=0, keepdims=True)

        w1_grad = np.dot(inputs.T, delta_hidden_layer)
        b1_grad = np.sum(delta_hidden_layer, axis=0, keepdims=True)

        w2_grad += 0.01 * self.weights[1]
        w1_grad += 0.01 * self.weights[0]

        self.weights[0] -= self.learning_rate * w1_grad
        self.biases[0] -= self.learning_rate * b1_grad

        self.weights[1] -= self.learning_rate * w2_grad
        self.biases[1] -= self.learning_rate * b2_grad

    def fit(self, inputs, outputs, clear = False):
        for epoch in range(self.epochs):
            it = 0
            while it < len(inputs):
                inputs_batch = inputs[it:it+self.batch_size]
                outputs_batch = outputs[it:it+self.batch_size]
                hidden_layer, output_layer = self.feed_forward(inputs_batch)

                loss = self.loss_function(output_layer, outputs_batch)
                self.loss.append(loss)

                self.back_propagate(inputs_batch, hidden_layer,
                                    output_layer, outputs_batch)

                if loss > 70:
                    loss_str = ("- - - - " + color.BOLD + "Epoch: {:d}/{:d}\t" + color.FAIL + "Loss: {:.2f}\t").format(
                        epoch+1, self.epochs, loss) + color.ENDC
                elif loss < 70 and loss > 10:
                    loss_str = ("- - - - " + color.BOLD + "Epoch: {:d}/{:d}\t" + color.WARNING + "Loss: {:.2f}\t").format(
                        epoch+1, self.epochs, loss) + color.ENDC
                elif loss <= 10 and loss > 1:
                    loss_str = ("- - - - " + color.BOLD + "Epoch: {:d}/{:d}\t" + color.OKGREEN + "Loss: {:.2f}\t").format(
                        epoch+1, self.epochs, loss) + color.ENDC

                epoch_prog, total_prog = self.get_progress(inputs, it)
                progress_str = color.BOLD + 'Epoch Progress : |' + color.OKBLUE + epoch_prog + color.ENDC + '|\t' + \
                    color.BOLD + 'Total Progress : |' + color.OKBLUE + \
                    total_prog + color.ENDC + color.BOLD + '|' + color.ENDC
                
                if clear:
                    time.sleep(0.001)
                    print(chr(27) + "[2J")

                print(loss_str + progress_str)

                

                it += self.batch_size

    def get_progress(self, inputs, it):
        epoch_bar = ''
        total_bar = ''
        total_it = len(inputs)/self.batch_size
        epoch_bar_ticks = (it / (total_it)) * 10
        total_bar_ticks = (it / (total_it * self.epochs)) * 5

        while epoch_bar_ticks > 0:
            epoch_bar += '█'
            epoch_bar_ticks -= 1

        while len(epoch_bar) < 10:
            epoch_bar += ' '

        while total_bar_ticks > 0:
            total_bar += '█'
            total_bar_ticks -= 1

        while len(total_bar) < 5:
            total_bar += ' '

        return epoch_bar, total_bar

    def predict(self, inputs):
        hidden_layer, prediction = self.feed_forward(inputs)
        return prediction

    def accuracy_test(self, inputs, result):
        prediction = self.predict(inputs)
        acc = float(np.sum(np.argmax(prediction, 1) == result)) / \
            float(len(result))
        if (acc*100) > 85:
            print(color.UNDERLINE + color.BOLD + '- - - - Test accuracy : ' +
                  color.OKGREEN + '{:.2f}% - - - -'.format(acc*100) + color.ENDC)
        elif (acc*100) < 50:
            print(color.UNDERLINE + color.BOLD + '- - - - Test accuracy : ' + color.FAIL +
                  '{:.2f}% - - - -'.format(acc*100) + color.ENDC)
        else:
            print(color.HEADER + color.BOLD + '- - - - Test accuracy : ' +
                  color.WARNING + '{:.2f}% - - - -'.format(acc*100) + color.ENDC)
