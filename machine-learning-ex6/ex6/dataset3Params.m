function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
bestAccuracy = 0;
possibleC = 0.01 * 2.^ [0:11];
possibleSigma = 0.01 * 2.^ [0:11];
numC = size(possibleC, 2);
numSig = size(possibleSigma, 2);
for i = 1:numC
  for j = 1:numSig
    thisC = possibleC(i);
    thisSig = possibleSigma(j);
    model = svmTrain(X, y, thisC, @(x1, x2) gaussianKernel(x1, x2, thisSig));
    predictions = svmPredict(model, Xval);
    accuracy = mean(double(predictions == yval));
    if accuracy > bestAccuracy
      C = thisC;
      sigma = thisSig;
      bestAccuracy = accuracy
      disp([C sigma]);
    endif
    disp(i*numC + j);
  endfor
endfor

[C, sigma];

% =========================================================================

end
