from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()
import numpy as np

def sigmoid(x):
 return 1/(1 + np.exp(-x))
m = len(train_X)
J = 0

theta= np.zeros(2)
grad = np.zeros(len(theta))
m ,ux,uy = np.shape(train_X)
train_X= np.reshape(train_X,(m,ux*uy))
# h = sigmoid(train_X*theta)
# print(h)
# J = -np.transpose(train_y)*log(h)- np.transpose(np.ones(m,1)-train_y)*(log(ones(m,1)-h))
# grad = np.transpose(train_X)*(h-y);

# theta(1) = 0;
# J = (J + (lambda/2)*theta'*theta)* (1/m);
# grad = (grad + lambda*theta).* (1/m);
