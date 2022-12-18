#! /usr/bin/env python
import numpy as np


class Softmax_Class():

    def __init__(self):

        pass

    def calculate_softmax(self, x_set):

        temp = np.exp(x_set) / np.sum(np.exp(x_set))
        print(temp)

        output = np.zeros_like(x_set)
        output[np.argmax(temp)] = 1

        return output

    def main(self, x_set):

        output = self.calculate_softmax(x_set)
        print(output)


if __name__ == "__main__":

    x_set = np.array([2, 0, 1])
    softmax = Softmax_Class()
    softmax.main(x_set)
