function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Iterate over each of the given 300 inputs
for(i=1:size(X)(1))
  % init X distance from cluster 
  dist = Inf;
  % for each cluster
  for(j=1:K)
    % find the distance of input X from each cluster
    %dist_from_centroid = (X(i,1) - centroids(j,1))**2 + (X(i,2) - centroids(j,2))**2;
    D = bsxfun(@minus, X(i,:), centroids(j,:));
    dist_from_centroid = sum(D.^2,2);
    % Assign  the input to closest cluster
    % ie, set idx(i) to closest cluster
    if(dist_from_centroid < dist)
      dist = dist_from_centroid;
      idx(i) = j;
    endif
  endfor
endfor

% =============================================================

end

