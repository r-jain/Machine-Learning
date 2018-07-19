function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% Execute Feed Formard propagation
% Add ones to the X data matrix for biased input in layer 1 (input layer)
X = [ones(m, 1) X];
% After transpose xi is a column vector in X (401x5000)
X =  X';

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
% ai is output column vector for xi input vector
A3 = zeros(size(z2));
A3 = A3 .+ 1./(1+e.^-(z2));


% Calculate Cost J(@) (Non Regularized)
% For each Class(k)
for k = 1:num_labels
  % Calculate and add the cost for each xi for given K
  for i= 1:m
    % Activation output of the kth output unit in the ai output vector for xi input vector 
    % Value in kth element of ith output vector
    hx_ik = A3(:,i)(k);
    % Value = 1; if calculated yi for xi input matches k value(class for iteration); 0 otherwise 
    y_ik = y(i) == k;

    % Evaluate cost for xi
    cost = y_ik*log(hx_ik);
    cost = cost + ((1-y_ik)*log(1-hx_ik));
    cost = cost*(-1/m);
    
    % Add Computed cost for xi for given K
    J = J+ cost;
 
  end;
    
end;

% Adjust cost for regularization
% Add @(ij) square valyes for input and hidden layers, Ignore the bias unit
J = J + ((lambda/(2*m)) * (sum(sum(Theta1(:,2:end).**2)) + sum(sum(Theta2(:,2:end).**2))))

% =========================================================================


% delta for output layer (l =3)  [ai - yi for given k] (10 x 5000)
deltas3 = zeros(size(A3));

% For each input Implment forward-prop/back-prop to compute deltas and then D1, D2 gradients
for i= 1:m
    % Evaluate delta for output layer : delta3
    % A vector for ith input xi (10 x 1)
    Ai = A3(:,i);
    
    % y vector element has value = 1 for the index value =(y(i)); All other values are 0  (10 x 1)
    Yi = zeros(size(A3)(1), 1);
    % here y(i) = output class for given xi
    Yi(y(i)) = 1;
    
    % Implement back propagation to evaluate Theta1_grad, Theta2_grad
    % D(i) = A(i) - Y(i) for given k
    delta = Ai .- Yi;
    deltas3(:,i) = deltas3(:,i) .+ delta;
    
end;

% Evaluate deltas for hidden layers 
deltas2 = (Theta2'*deltas3);
% remove  d2(0) : (26 x 5000) => (25 x 5000)
deltas2 = deltas2(2:end, :);
deltas2 = deltas2.*sigmoidGradient(z1);

% Evaluate Theta1_grad, Theta2_grad : D(L) = (1/m)* (deltas(L+1)*A(L)')
%(25 x 5000) * (5000 x 401) => Theta1_grad (25 x 401)
Theta1_grad = (1/m)*deltas2*X'; 
%(10 x 5000) * (5000 x 26) => Theta2_grad(10 x 26)
Theta2_grad = (1/m)*deltas3*X2'; 

% apply relarization on Gradients (other than for bias tehta values)
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) .+ (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) .+ (lambda/m)*Theta2(:,2:end)

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
