function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% For each centroid
for(i=1:K)
  % iterate over each input
  centroid_mean = zeros(size(1,n));
  count_asigned = 0;
  for(j=1:m)
    % evaluate mean of assigned inputs for given cetroid
    if(idx(j) == i)
      centroid_mean = centroid_mean + X(j,:);
      count_asigned = count_asigned +1;
    endif
  endfor
  centroid_mean = centroid_mean/count_asigned;
  centroids(i,:) = centroid_mean;
endfor

% =============================================================


end

