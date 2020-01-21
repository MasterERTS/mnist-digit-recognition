import numpy as np 

class NeuralNetwork():
    def __init__(self, n_layers, n_nodes):
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        self.layers = []
