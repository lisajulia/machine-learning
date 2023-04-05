from keras.datasets import mnist
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


(train_X, train_y), (test_X, test_y) = mnist.load_data()

# grad = np.zeros(len(theta))
m, ux, uy = np.shape(train_X)
n = ux * uy
train_X = np.reshape(train_X, (m, n))
# theta = np.zeros(10, n)

from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()
logisticRegr.fit(train_X, train_y)

testVec =np.reshape(test_X[0], (n, 1))
res = logisticRegr.predict(testVec)
print(testVec)
print(res)


# h = sigmoid(train_X*theta)
# print(h)
# J = -np.transpose(train_y)*log(h)- np.transpose(np.ones(m,1)-train_y)*(log(ones(m,1)-h))
# grad = np.transpose(train_X)*(h-y);

# theta(1) = 0;
# J = (J + (lambda/2)*theta'*theta)* (1/m);
# grad = (grad + lambda*theta).* (1/m);
