function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    theta_new = theta;
    for j = 1:columns(X)
      correction_factor = 0;
      for i = 1:m
        correction_factor = correction_factor + (X(i,:) * theta - y(i))*X(i,j);
      endfor
      correction_factor = correction_factor*alpha/m;
      theta_new(j) = theta(j) - correction_factor;
    endfor
    theta = theta_new;   
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
