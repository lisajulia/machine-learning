function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); %number of parameters we want to find

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

for i = 1:m
   h = sigmoid(theta'*X(i,:)');
   J = J - y(i)*log(h) - (1 - y(i))*log(1 - h);
   for j = 1:n
     grad(j) = grad(j) + (h - y(i))*X(i,j);
   endfor
endfor
J = J * (1/m);
grad = grad.* (1/m);

end
