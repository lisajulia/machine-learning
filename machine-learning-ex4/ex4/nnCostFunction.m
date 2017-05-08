function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
        
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Calculate the matrix h
z2 = Theta1*[ones(m,1),X]';
a2 = [ones(m,1),sigmoid(z2)'];
z3 = Theta2*[a2]';
h = sigmoid(z3)';

% Transform y into a num_labels x m matrix
Y = zeros(m, num_labels);
for i = 1:m
  Y(i,y(i)) = 1;
end

J = sum(sum(- Y.*log(h) - (ones(size(Y)) - Y).*log(ones(size(h)) - h)))/m;

% add regularization term
Theta1_ = Theta1;
Theta1_(:,1) = zeros(size(Theta1,1),1);

Theta2_ = Theta2;
Theta2_(:,1) = zeros(size(Theta2,1), 1);

J = J + lambda*(sum(sum(Theta1_.^2)) + sum(sum(Theta2_.^2)))/2/m;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
theta_3 = h - Y;

theta_2 = (theta_3*Theta2)' .* [zeros(1,m); sigmoidGradient(z2)];
%remove the bias unitq
theta_2 = theta_2(2:end,:);

Lambda_2 = (a2'*theta_3)';
Lambda_1 = theta_2*[ones(m,1),X];

Theta2_grad = Lambda_2./m + lambda*Theta2_/m;
Theta1_grad = Lambda_1./m + lambda*Theta1_/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
