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
err = 1;
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

% try all the combinations of C and sigma
vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
for(i=1:size(vals)(1))
  current_C = vals(i);
  for(j=1:size(vals)(1))
    current_sigma = vals(j);   
    % Train the SVM for give c and sigma to obtain model
    model= svmTrain(X, y, current_C, @(x1, x2) gaussianKernel(x1, x2, current_sigma));
    % predict values using the trained model
    predictions = svmPredict(model, Xval);
    % Compute prediction error
    current_err = mean(double(predictions ~= yval));
    if(current_err <err)
      err = current_err;
      C = current_C;
      sigma = current_sigma;
      %fprintf("Current_error %d%, current_C %d%, current_sigma %d%", current_err, current_C, current_sigma)
    endif
  endfor
endfor
% =========================================================================

end
