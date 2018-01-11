import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_incorrect_classes = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      # note delta = 1
      margin = scores[j] - correct_class_score + 1
      if margin > 0:
        num_incorrect_classes += 1
        # gradient update for incorrect rows
        dW[:, j] += X[i]
        loss += margin
    # gradient update for correct rows
    dW[:, y[i]] += -num_incorrect_classes * X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg * W
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  delta = 1.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(X.shape[0]), y]
  # calculate the margins using a vectorized operation
  margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + delta)
  # correct margin for correct label where the margin is currently delta
  margins[np.arange(X.shape[0]), y] = 0

  loss = np.sum(margins)
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)

  # start calculating dW
  X_mask = np.zeros(margins.shape)
  X_mask[margins > 0] = 1
  # create an array of the number of incorrect guesses for each class
  incorrect_classes_for_sample = np.sum(X_mask, axis=1)

  # coefficients of gradient correction for correct classes
  X_mask[np.arange(X.shape[0]), y] = -incorrect_classes_for_sample
  # convert the mathematics of the loop to matrix multiplication
  dW = X.T.dot(X_mask)
  dW /= X.shape[0]
  dW += reg * W
  return loss, dW
