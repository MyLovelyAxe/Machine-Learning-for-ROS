import numpy as np
import matplotlib.pyplot as plt


def create_data_set(points, classes):

    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4,
                        points) + np.random.randn(points)*0.5
        X[ix] = np.c_[r*np.sin(t*1.0), r*np.cos(t*1.0)]
        y[ix] = class_number
    return X, y


# input X shape: (800,2)
# target y shape: (800,)
X, y = create_data_set(400, 2)


N = 400  # number of points per class
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
  # score is the values before softmax process, a.k.a 'digits' in course book
  # in this case, the activation function of hidden layer is ReLU, the activation function of output layer is softmax
    hidden_layer = np.maximum(0, np.dot(X, W) + b)  # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2

  # compute the Softmax class probabilities
    exp_scores = np.exp(scores)
  # the np.sum() is operated along the axis of '1'
  # which means that in 'score' which is of shape (800,2),
  # every row takes addition, e.g. 0 row, [0,0] + [0,1]
  # then the output of np.sum() will be of shape of (800,), containing 800 summation for 800 pair points
  # 'score' and 'exp_score' are of the same shape of (800,2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

  # compute the loss (cross-entropy loss)
  # the explaination of what is "probs[range(num_examples), y]" doing:
  # due to the fact that in each data 'pair', there are only 2 points, a.k.a only x1 and x2
  # probs[range(800),y] does this:
  # index each row according to range(800) which starts from 0 to 800 with step 1
  # (we can assume there is an 'i' here, like: for i in range(800))
  # and pick the value according to the correspongding position, which is determined by y[i]
  # if y[i] == 0, then keep the 0th colomn, if y[i] == 1, then keep the 1th colomn
  # this process is similar with:
  # when y[i] == False(0), then delete it, when y[i] == True(1), then keep it
  # and according to the formular of cross-entropy:
  #     h(x) = -log(e) if y == 1
  #     h(x) = -log(1-e) if y == 0
  # let us come back to previous step: compute softmax
  # because there are only 2 parts in one input pair, x1 and x2
  # after softmax, there come e1 and e2 ps: e1 = exp(x1)/(exp(x1)+exp(x2)), e2 = exp(x2)/(exp(x1)+exp(x2))
  # and e1 = 1 - e2
  # so this situation can correspond with h(x)
  # if y == 1, h(x) = -log(e), resembling that h(x) = -log(e1)
  # if y == 0, h(x) = -log(1-e), resembling that h(x) = -log(1-e1) = -log(e2)
  # in conclusion:
  # this line selects corresponding probability according the class label, which is the same with cross-entropy formular
  # but the situation only suits when each input only has 2 parts
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)/num_examples
    #reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss  # + reg_loss
    if i % 100 == 0:
        print "i:", i, "loss: ", loss
        ii.append(i)
        error.append(loss)

  # compute the gradient on scores
  # this actually calculate:
  # d(cross entropy loss) / d(scores)
  # "scores" is the value before softmax
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

  # apply the backpropate alghorithm(BP) the gradient

  # first apply into parameters W2 and b2
  # dw2 = d(scores)/dw2 * d(cross entropy loss)/d(scores)
  # because scores = hidden * w2 + b2
  # d(scores)/dw2 = hidden
  # and previously we have calculated that d(cross entropy loss)/d(scores) = dscores
  # then dW2 = hidden * dscores
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
  # next BP into hidden layer
  # because scores = hidden * w2 + b2
  # and previously we have calculated that d(cross entropy loss)/d(scores) = dscores
  # then dhidden = w2 * dscores
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
