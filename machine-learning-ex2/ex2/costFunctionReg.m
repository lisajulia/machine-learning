function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
[J,grad] = costFunction(theta, X, y);

for j = 2:n
  J = J + (lambda/(2*m))*theta(j)^2;
  grad(j) = grad(j) + lambda/m*theta(j);
endfor

end
