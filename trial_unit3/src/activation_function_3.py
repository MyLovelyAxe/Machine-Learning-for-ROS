"""
This is a complete solution for exercise 3.1
"""

import numpy as np


X = np.array(([1, 8], [2, 9], [3, 10]), dtype=float)


class NeuralNetwork(object):
    def __init__(self):
        # Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights and biases (weight are randomly initalized. Biases are initialized to one)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.B1 = np.ones((1, self.hiddenLayerSize))
        self.B2 = np.ones((1, self.outputLayerSize))

    def forwardPropagation(self, X):
        # Definitioon of signals (outputs) on step of NN.
        self.z2 = np.dot(X, self.W1) + self.B1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2) + self.B2
        y = self.sigmoid(self.z3)
        return y

    def sigmoid(self, z):
        # Apply sigmoid activation
        return 1/(1+np.exp(-z))


NN = NeuralNetwork()
y = NN.forwardPropagation(X)
print(y)
