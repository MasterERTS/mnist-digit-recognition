import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.max(z, 0)


def softmax(z):
    exp = np.exp(z)
    sm = exp/np.sum(exp, axis=1, keepdims=True)
    return sm


def cross_entropy(z, out):
    i = np.argmax(out, axis=1).astype(int)
    probability = z[np.arange(len(z)), i]
    log = np.log(probability)
    loss = -1.0 * np.sum(log) / len(log)
    return loss


def regularization(la, w1, w2):
    w1_loss = 0.5 * la * np.sum(w1**2)
    w2_loss = 0.5 * la * np.sum(w2**2)
    return w1_loss + w2_loss
