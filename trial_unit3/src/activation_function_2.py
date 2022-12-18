#! /usr/bin/env python
import numpy as np


class Simple_NN_Class():

    def __init__(self):

        # the sizes of different layers
        # attention:
        #   input size here is the size of 'single one' input, not input dataset
        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1

    def sigmoid(self, z):

        return 1/(1+np.exp(-z))

    def forwardPropagation(self, input):

        # input layer -- hidden layer
        # input:    (3,2)
        # w1:       (2,3)
        # b1:       (3,)
        # z2 = input * w1 + b1: (3,)
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.ones(self.hidden_size)
        z2 = np.dot(input, self.w1) + self.b1
        a2 = self.sigmoid(z2)

        # hidden layer -- output layer
        # z2:   (3,)
        # w2:   (3,1)
        # b2:   (1,)
        # z2 = z2 * w2 + b2: (1,)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.ones(self.output_size)
        z3 = np.dot(a2, self.w2) + self.b2
        a3 = self.sigmoid(z3)

        return a3


if __name__ == "__main__":

    input = np.array(([1, 8], [2, 9], [3, 10]), dtype=float)
    simple_NN = Simple_NN_Class()
    output = simple_NN.forwardPropagation(input)
    print(output)
