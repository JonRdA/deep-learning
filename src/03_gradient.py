# https://www.youtube.com/watch?v=-_A_AAxqzCg

import numpy as np
import scipy as sc

import matplotlib.pyplot as plt


# Declare function
func = lambda th: np.sin(1 / 2 *th[0] ** 2 - 1 / 4 * th[1] ** 2 + 3) * \
    np.cos(2 *th[0] + 1 - np.e ** th[1])

# Test function
val1 = func([5, 3])
#print(val1)

res = 20       # resolution of function

# Generate values
_X = np.linspace(-2, 2, res)
_Y = np.linspace(-2, 2, res)

# Empty array to hold result values
_Z = np.zeros((res, res))

# Calculate function and populate _Z
for ix, x in enumerate(_X):
    for iy, y in enumerate(_Y):
        _Z[iy, ix] = func([x, y])

#print(_Z)

plt.contourf(_X, _Y, _Z, res)
plt.colorbar()

# Random point on surface
theta = np.random.rand(2) * 4 - 2
plt.plot(theta[0], theta[1], "o", c="white")

_T = np.copy(theta)
h = .001        # diferential for derivative
lr = .01         # learning rate

grad = np.zeros(2)

for i in range(10000):
    for it, th in enumerate(theta):
        _T = np.copy(theta)
        _T[it] = _T[it] + h
        deriv = (func(_T) - func(theta)) / h
        grad[it] = deriv
    theta = theta - grad * lr
    if (i % 10) == 0:
        plt.plot(theta[0], theta[1], "o", c="red")
        
    print(func(theta))

plt.plot(theta[0], theta[1], "o", c="green")
plt.show()

