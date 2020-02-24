#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from lib.neural_network import NeuralNetwork

def test_or_gate():
    expected_inputs = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

    expected_outputs = np.array([[0, 1, 1, 1]]).T

    layers = [2, 1, 1]
    epochs = 5
    batch_size = 1
    learning_rate = 0.1

    nn = NeuralNetwork(layers, batch_size, epochs, learning_rate)

    nn.fit(expected_inputs, expected_outputs)

