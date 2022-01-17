"""
Linear regression and minimum squared code as explained by Dot CSV
https://www.youtube.com/watch?v=w2RJ1D6kz-o
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# Load dataset
boston = load_boston()
#print(type(boston))

X = boston.data[:, 5]
Y = boston.target

plt.scatter(X, Y, alpha=.3)

# Add column of ones for constant values.
X = np.array([np.ones(506), X]).T

B = np.linalg.inv(X.T @ X) @ X.T @ Y
print(B)

#plt.plot([4,9], [4 * B[1] + B[0], B[0] + B[1] * 9], c="red")
#plt.show()

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
