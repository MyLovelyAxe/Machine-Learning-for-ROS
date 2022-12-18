from __future__ import division
import numpy as np
import future
import matplotlib.pyplot as plt
import os

################################
###### create sample data ######
################################


def create_sample_data_set(size, outer_width, outer_height, a, b, inner_width, inner_height):

    corners = np.array([[0, 0], [a, 0], [a+inner_width, 0],
                        [0, b], [a+inner_width, b],
                        [0, b+inner_height], [a, b+inner_height], [a+inner_width, b+inner_height]])
    top_height = outer_height - (b + inner_height)
    right_width = outer_width - (a + inner_width)
    widths = np.array([a, inner_width, right_width, a,
                      right_width, a, inner_width, right_width])
    heights = np.array([b, b, b, inner_height, inner_height,
                       top_height, top_height, top_height])

    areas = widths * heights
    shapes = np.column_stack((widths, heights))

    regions = np.random.multinomial(size, areas/areas.sum())
    indices = np.repeat(range(8), regions)
    unit_coords = np.random.random(size=(size, 2))
    pts = unit_coords * shapes[indices] + corners[indices]

    # give the labels (supervise machine learnining) to the classes => 0 (robot free to go) or 1 (prohibited)

    y_out = np.ones(size, dtype='uint8')
    i = 0

    for pt in pts:

        if pt[0] > (5/7) or pt[0] < (3/7) or pt[1] > (3/4) or pt[1] < (1/4):
            y_out[i] = 0

        i = i + 1

    return pts, y_out


plt.figure(figsize=(8, 8))
# X: input data (2000, 2)
# y: class (2000,)
X, y = create_sample_data_set(2000, 1, 1, 0.01, 0.01, 0, 0)
print("input shape: ", X.shape)
print("class shape: ", y.shape)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()

#######################
###### trainging ######
#######################

D = 2  # dimensionality
K = 2  # number of classes
h = 1000  # size of hidden layer
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))
step_size = 1e-0


num_examples = X.shape[0]
ii = []
error = []

# here in for-loop we deply the back propagation alghorithm (gradient descent)

for i in range(3000):

  # evaluate class scores, [N x K]

    hidden_layer = np.maximum(0, np.dot(X, W) + b)  # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2

  # compute the Softmax class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

  # compute the loss (cross-entropy loss)
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)/num_examples
    #reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss  # + reg_loss
    if i % 100 == 0:
        print "i:", i, "loss: ", loss
        ii.append(i)
        error.append(loss)

  # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

  # apply the backpropate alghorithm(BP) the gradient
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
  # next BP into hidden layer
    dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
  # finally into W,b
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

  # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

# show results
h = 0.03
x_min, x_max = X[:, 0].min() - 0, X[:, 0].max() + 0
y_min, y_max = X[:, 1].min() - 0, X[:, 1].max() + 0
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
# forward propagation
Z = np.dot(np.maximum(
    0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
# take larger digit
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

# show training process
plt.plot(ii, error, label="error")
plt.xlabel("epoch", fontsize=16)
plt.ylabel("loss function", fontsize=16)
plt.title("training process", fontsize=18)
plt.legend()
plt.show()

# show accuracy
test_hidden = np.maximum(0, np.dot(X, W) + b)
test_scores = np.dot(test_hidden, W2) + b2
test_class = np.argmax(test_scores, axis=1)
print("the accuracy is: ", np.mean(test_class == y))

###############################
###### export parameters ######
###############################

export_path = "/home/user/catkin_ws/src/weights"

try:
    os.mkdir(export_path)
except:
    print("folder already exists")
    pass

np.save(export_path + "/w.npy", W)
np.save(export_path + "/b.npy", b)
np.save(export_path + "/w2.npy", W2)
np.save(export_path + "/b2.npy", b2)
