import numpy as np
import lib.maths_util as mathlib


class NeuralNetwork():
    def __init__(self, layers, batch_size, epochs, learning_rate):
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.weights = []
        self.biases = []
        self.loss = []
        n_layers = len(layers)

        for i in range(n_layers - 1):
            self.weights.append(np.random.normal(
                0, 1, [self.layers[i], self.layers[i+1]]))
            self.biases.append(np.zeros((1, self.layers[i+1])))

    def feed_forward(self, inputs):
        inputs = inputs.astype(float)

        layer0 = inputs
        layer1 = mathlib.relu(np.dot(layer0, self.weights[0]) + self.biases[0])
        layer2 = mathlib.softmax(
            np.dot(layer1, self.weights[1]) + self.biases[1])

        return layer1, layer2

    def loss_function(self, predicted_outputs, outputs):
        loss = mathlib.cross_entropy(predicted_outputs, outputs)
        loss += mathlib.regularization(0.01, self.weights[0], self.weights[1])

        return loss

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

    def fit(self, inputs, outputs):
        for epoch in range(self.epochs):
            hidden_layer, output_layer = self.feed_forward(inputs)
            loss = self.loss_function(output_layer, outputs)
            self.loss.append(loss)
            self.back_propagate(inputs, hidden_layer, output_layer, outputs)

    def predict(self, inputs):
        prediction = self.feed_forward(inputs)
        print("Result to your output is === " + prediction)

    def accuracy_test(self, inputs, result):
        prediction = self.feed_forward(inputs)
        acc = float(np.sum(np.argmax(prediction, 1) == result)) / \
            float(len(result))
        print('Test accuracy : {:.2f}%').format(acc*100)
