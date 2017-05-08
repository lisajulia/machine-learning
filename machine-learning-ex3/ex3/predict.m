function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

%add the ones to the matrix X
a1 = [ones(m,1),X];

z1 = a1*Theta1';

a2 = [ones(m,1),sigmoid(z1)];

z2 = a2*Theta2';

a3 = sigmoid(z2);

[value, p] = max(a3,[],2);

end
