function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    h_theta = @(x) theta' * x';
    
    results1 = zeros(length(X), 1);
    results2 = zeros(length(X), 1);

    for row = 1:length(X)
        results1(row) = h_theta(X(row, :)) - y(row);
        results2(row) = (h_theta(X(row, :)) - y(row)) * X(row, 2);
    end
    
    temp1 = theta(1) - alpha * (sum(results1) / m);
    temp2 = theta(2) - alpha * (sum(results2) / m);
    theta(1) = temp1;
    theta(2) = temp2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
