function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

pos = y == 1; % get indexes of admission
neg = y == 0; % get indexes of rejection

% get exam scores for acceptance and rejection
examOnePos = X(pos, 1);
examOneNeg = X(neg, 1);
examTwoPos = X(pos, 2);
examTwoNeg = X(neg, 2);

plot(examOnePos, examTwoPos, 'rx', 'MarkerSize', 10);
plot(examOneNeg, examTwoNeg);

% =========================================================================
hold off;

end
