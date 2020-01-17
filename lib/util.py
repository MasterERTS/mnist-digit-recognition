import numpy as np

def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))