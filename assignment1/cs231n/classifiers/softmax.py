import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
      # compute scores
      scores = X[i].dot(W)
      # this is a trick to minimise computational instability
      scores -= np.max(scores)
      # extract some information for ease
      correct_label = y[i]
      correct_label_score = scores[correct_label]
      # calculate denominator in the softmax loss calculation
      sum_j = np.sum(np.exp(scores))
      # define a probability function
      p = lambda x: np.exp(x) / sum_j
      loss += -np.log(p(correct_label_score))
      for j in range(num_classes):
        p_j = p(scores[j])
        dW[:, j] += (p_j - (j == correct_label)) * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  scores = X.dot(W)
  num_train = X.shape[0]
  # trick to minimise computational instability
  scores -= np.max(scores, axis=1, keepdims=True)
  sum_j_list = np.sum(np.exp(scores), axis=1, keepdims=True)
  probability_list = np.exp(scores) / sum_j_list
  correct_probability_list = probability_list[np.arange(num_train), y]
  loss = np.sum(-np.log(correct_probability_list))
  loss /= num_train
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
