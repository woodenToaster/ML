function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h_theta = @(x) sigmoid(dot(theta',x));
results = zeros(m, 1);
for row = 1:m
    results(row) = -y(row) * log(h_theta(X(row, :))) .- (1 - y(row)) * log(1 - h_theta(X(row, :)));
end
J = (1 / m) * sum(results);

results = zeros(m, length(theta));

for row = 1:m
    for j = 1:size(X,2)
      results(row, j) = ((h_theta(X(row,:)) - y(row)) * X(row,j));
    end
end

for col = 1:size(X,2)
    grad(col) = sum(results(:,col)) ./ m;
end 
% =============================================================

end
