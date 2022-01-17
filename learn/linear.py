"""Various attempts at solving multivariate linear regression with
gradient descent algorithm."""

import numpy as np

## Multi variable linear regression with matrices.
#
#n = 10     # number of values, instances
#p = 3       # number of variables
#lr = .0001
#
## Create random numbers between 0 & 100 and add column of ones for constant.
#x = np.random.random((n,p)) * 100
#X = np.insert(x, 0, np.ones(n), axis=1)
#
## Create variable Y with values of X
#Y = 37 * X[:, 0] + 11 * X[:,1] - 12 * X[:,2] + 7 * X[:,3]# + .00005*np.random.random(n)
#
##b = np.linalg.inv(X.T @ X) @ X.T @ Y
#
##Y_ = X @ b
#
##plt.plot(X[:, 3], Y, "ro")
##plt.show()
#
##Initialize random parameters vector.
#theta = np.ones(p + 1)
#theta = (np.random.random(p + 1) -  0.5) * 200
##T = np.array([37, 11, -12, 7.01])
#
#epochs = 100000
#err = []
#
#for i in range(epochs):
#    # Calculate error of out prediction
#    y_pred = X @ theta
#    error = y_pred - Y
#    mse = (error**2).sum()
#    gradients = 2/n * X.T @ error
#    theta = theta - lr * gradients
#    if i % 1000 == 0:
#        print(theta, mse)
#        err.append(mse)
#    if i == 8000:
#        print("here")
#        lr /=10
#
#plt.plot(err[4:])
#plt.show()
#
# ------------------------------------------------------------------------------
# Gradient descent coded in tensorflow.

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

fetched_data = fetch_california_housing()
m, n = fetched_data.data.shape
data_with_bias = np.c_[np.ones((m, 1)), fetched_data.data]

X = tf.constant(data_with_bias, dtype=tf.float32, name="features")

y = tf.constant(fetched_data.target.reshape(-1, 1), dtype=tf.float32, name="target")

XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(XT, X)), XT), y)
print(theta)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_with_bias)

n_epoch = 1000
learning_rate = .01

X = tf.constant(scaled_data, dtype=tf.float32, name="scaled")
theta = tf.Variable(tf.random.uniform([n+1, 1], -1, 1), name="theta")

for epoch in range(n_epoch):
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.matmul(tf.transpose(error), error) / m
    gradients = 2/m * tf.matmul(tf.transpose(X), error)
    theta = theta - learning_rate * gradients

    if epoch % 100 == 0:
        print("Epoch: ", epoch, "MSE: ", mse)
