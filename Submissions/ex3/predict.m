function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================
% Add ones to the X data matrix (5000x400) =>(5000x401) [Add biased input unit]
X = [ones(m, 1) X];

% Convert X(5000 x 401) to X'(401x5000) each column is now xi (xi is column vector in X)
X = X';

% Evaluate Z1 (25x5000)
z1 = Theta1*X;

% Evaluate hx1 = g(z1) =>X2/A2 (activation units in hidden layer) (25x5000)
X2 = zeros(size(z1));
X2 = X2 .+ 1./(1+e.^-(z1));

% Convert to (5000 x 25)
X2 = X2';
% Add ones to the X2 data matrix[Add biased input unit] (5000 x 26)
X2 = [ones(m, 1) X2];
% convet to (26 x 5000) with each column vector = x2i
X2 = X2';

% Evaluate z2 (10x5000)
z2 = Theta2*X2;

% Evaluate hx2 = g(z2) =>A3 (10x5000)
A3 = zeros(size(z2));
A3 = A3 .+ 1./(1+e.^-(z2));

% assign the index of value with highest g(z) value as predicted class
% max gives the row vector  (1x5000) with max value (matching class/row-num) from each column
[x, ix] = max(A3);

% Assign iX' to p (5000x1)
p = ix';

end
