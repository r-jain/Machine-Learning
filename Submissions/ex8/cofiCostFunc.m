function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% for each user (Theta transpose-row index)
for(j=1:num_users)
  % for each movie (X transpose-row index)
  for(i=1:num_movies)
    % Add cost where R(i,j) =1. (only the movies which are rated)
    J = J + ((Theta(j,:)*X(i,:)') - Y(i,j))**2 * R(i,j);
  endfor
endfor
J = J/2;
regularization_cost = (lambda/2)*(sum(sum(X.**2)) + sum(sum(Theta.**2)));
J = J + regularization_cost;


% claculate X_grad
for(i=1:num_movies)
  % list of all the users that have rated movie i
  idx = find(R(i, :)==1);
  
  % Temporary matrices for set of users which have rated the movie i
  Theta_temp = Theta(idx, :);
  Y_temp = Y(i, idx);
  
  % x gradient row vector
  X_grad(i, :) = (X(i, :)*Theta_temp' - Y_temp)*Theta_temp;
  
  % X grad regularization
  X_grad(i, :) = X_grad(i, :) + lambda*(X(i,:));
  
endfor

% claculate Theta_grad
for(j=1:num_users)
  % list of all the movies which are rated by user j
  idx = find(R(:, j)==1);
  
  % Temporary matrices for set of movies which are rated by user j
  X_temp = X(idx, :);
  Y_temp = Y(idx, j);
  
  % Theta gradient row vector
  Theta_grad(j, :) = (X_temp*Theta(j, :)' - Y_temp)'*X_temp;
  
  % Theta grad regularization
  Theta_grad(j, :) = Theta_grad(j, :) + lambda*(Theta(j,:));
  
endfor

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
