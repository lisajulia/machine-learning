function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));
h = sigmoid(X*theta); %in 1xm

J = -y'*log(h)- (ones(m,1)-y)'*(log(ones(m,1)-h));
grad = X'*(h-y);

theta(1) = 0;
J = (J + (lambda/2)*theta'*theta)* (1/m);
grad = (grad + lambda*theta).* (1/m);

end
