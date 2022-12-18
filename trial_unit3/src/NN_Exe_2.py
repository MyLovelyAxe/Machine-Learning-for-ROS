import numpy as np


def load_weights():
    b = np.load("/home/user/catkin_ws/src/weights/b.npy")
    b2 = np.load("/home/user/catkin_ws/src/weights/b2.npy")
    W = np.load("/home/user/catkin_ws/src/weights/w.npy")
    W2 = np.load("/home/user/catkin_ws/src/weights/w2.npy")
    return b, b2, W, W2


def prediction(XX, b, b2, W, W2):
    hidden_layer = np.maximum(0, np.dot(XX, W)+b)
    scores = np.dot(hidden_layer, W2) + b2
    predict = np.argmax(scores, axis=1)
    return int(predict)


b, b2, W, W2 = load_weights()
XX = np.array([0.6, 0.4])
predict = prediction(XX, b, b2, W, W2)
print("prediction of position [{}, {}] is: {}".format(XX[0], XX[1], predict))
XX = np.array([0.25, 0.6])
predict = prediction(XX, b, b2, W, W2)
print("prediction of position [{}, {}] is: {}".format(XX[0], XX[1], predict))
