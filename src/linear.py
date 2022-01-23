"""
Multivariate linear regression solved with OLS and gradient descent.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Geneate initial dataset.
n = 5      # variables
m = 10     # instances

X = tf.ones([m, 1])
Y = X

for i in range(1, n + 1):
    # Generate uniform column value & add to X.
    x_col = tf.random.uniform([m, 1])
    X = tf.concat([X, x_col], axis=1)
    
    # Add column with factor to Y values.
    Y = tf.add(Y, x_col * i)

# Solve with Ordinary Least Squared (OLS) estimator.
part1 = tf.matmul(tf.transpose(X), X)
part2 = tf.matmul(tf.transpose(X), Y)
theta = tf.matmul(tf.linalg.inv(part1), part2)
print(theta)

# Solve with gradient descent algorithm

# Cost function
def cost(theta, X, Y, gradient=False):
    """Obtain cost function or derivative for point `theta`.
    J(theta) = 1/2 * (h0 - Y)^2
    Keyword argument selects derivative or mean squared error value."""
    h0 = tf.matmul(X, theta)
    err = tf.subtract(h0, Y)
    if gradient:
        return tf.matmul(tf.transpose(X), err)
    mse = tf.square(err)
    return tf.divide(mse, 2)

def track(res, i, theta, Y):
    """Track cost value of iteration on list `costs`."""
    mse = tf.reduce_mean(cost(theta, X, Y), axis=0)
    res[0].append(i)
    res[1].append(mse)

# Gradient descent
lr = .01                # learning rate
it = 10000              # iterations
tfr = it // 10          # track frequency

def train(X, Y, lr, it, tfr):
    """ Function to easily perform a gradient descent with some hyperparameters.
    Train multivariate linear regression with gradient descent, tracking"""
    # Generate random parameters, initial solution.
    theta = tf.random.uniform([n + 1, 1])
    res = [[], []]          # error tracking list
    
    for i in range(it):
        grd = cost(theta, X, Y, gradient=True)
        theta = theta - grd * lr
        if (i % tfr) == 0:
            track(res, i, theta, Y)

    return res

# Gradient descent
lr = .01                # learning rate
it = 10000              # iterations
tfr = it // 10          # track frequency

lrates = [.08, .05, .02, .01, .005, .001]
errors = []

#for lr in lrates:
#    res = train(X, Y, lr, it, tfr)
#    errors.append(res[1][-1])

#plt.plot(lrates, errors, "ro-")
#plt.show()

res = train (X, Y, .01, it, tfr)
[tf.print(i) for i in res[1]]
plt.plot(res[0], res[1])
plt.show()
