import numpy as np 

class NeuralNetwork():
    def __init__(self, layers, batch_size, epochs, learning_rate):
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def forward_propagate(self, inputs):
        pass

    def loss_function(self, inputs):
        pass

    def back_propagate(self, output, target):
        pass

    def gradiant_descent(self):
        pass

    def fit(self, inputs, outputs):
        pass

    def test(self, inputs, outputs):
        pass
