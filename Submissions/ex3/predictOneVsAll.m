function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       







% =========================================================================

% X(5000x401) (each row is input xi) 
% take transpose of current X to get actual X 
% X is now (401 x 5000) (each column is xi now)
X = X';

% all_theta (10x401) (each row is theta [] for a separate class (class = row index)
% theta is already in transpose format since each row is theta value for a class) 
% compute z (10x5000) = theta'X
z = all_theta*X;

% Calculate sigmoid value  g(z)/probability for each z value
% hx (10x5000) have each row with z vlaues for all xi(5000) for that class(row index)
hx = zeros(size(z));
hx = hx .+ 1./(1+e.^-(z));

% assign the index of value with highest g(z) value as predicted class
% max gives the row vector  (1x5000) with max value (matching class/row-num) from each column
[x, ix] = max(hx);

% Assign iX' to p (5000x1)
p = ix';

end
