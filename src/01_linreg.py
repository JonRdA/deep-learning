"""
Linear regression and minimum squared code as explained by Dot CSV
https://www.youtube.com/watch?v=w2RJ1D6kz-o
"""
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
#
## Load dataset
#boston = load_boston()
##print(type(boston))
#
#X = boston.data[:, 5]
#Y = boston.target
#
#plt.scatter(X, Y, alpha=.3)
#
## Add column of ones for constant values.
#X = np.array([np.ones(506), X]).T
#
#B = np.linalg.inv(X.T @ X) @ X.T @ Y
#print(B)
#
##plt.plot([4,9], [4 * B[1] + B[0], B[0] + B[1] * 9], c="red")
##plt.show()
#
## ------------------------------------------------------------------------------
## Gradient descent to linear model with normal variables
#x = X[:,1]
#y = np.array(Y)
##plt.scatter(x, y, alpha=.3)
##plt.show()
#
#def loss(m, c):
#    """Loss function, mean squared error."""
#    return ((y - (m*x + c))**2).mean()
#
#def loss_m(m, c):
#    """Partial derivative with respect to `m` of loss function."""
#    return -2 * (x * (y - m * x - c)).mean()
#
#def loss_c(m, c):
#    """Partial derivative with respect to `c` of loss function."""
#    return -2 * (y - m * x - c ).mean()
#
#
## Initialize parameters
#m = 0
#c = 0
#r = .01     # learning rate
#
#n = 100000
#l = np.zeros(n)
#
#for i in range(n):
#    lm = loss_m(m, c)
#    lc = loss_c(m, c)
#    #print(lm, lc, m, c)
#    m -= r * lm
#    c -= r * lc
#    l[i] = loss(m, c)
#    if i % 50000 == 0:
#        plt.scatter(x, y, alpha=.3, c="r")
#        plt.plot([4, 9], [4 * m + c, 9 * m + c], c="blue")
#        print([4, 9], [4 * m + c, 9 * m + c])
#        plt.show()
#
#print(m, c)
#
# -----------------------------------------------------------------------------
# Multi variable linear regression with matrices.

n = 10     # number of values, instances
p = 3       # number of variables

# Create random numbers between 0 & 100 and add column of ones for constant.
x = np.random.random((n,p)) * 100
X = np.insert(x, 0, np.ones(n), axis=1)

# Create variable Y with values of X
Y = 37 * X[:, 0] + 11 * X[:,1] - 12 * X[:,2] + 7 * X[:,3]# + .00005*np.random.random(n)

#b = np.linalg.inv(X.T @ X) @ X.T @ Y

#Y_ = X @ b

#plt.plot(X[:, 3], Y, "ro")
#plt.show()

#Initialize random parameters vector.
T = np.ones(p + 1)
T = np.array([37, 11, -12, 7.01])

# Calculate error of out prediction
Y_ = X @ T
E = (Y - Y_)**2 @ np.ones(n)
print(E)



# TODO calculate partial derivatives of losss func & implement matrix gradient
# descent
