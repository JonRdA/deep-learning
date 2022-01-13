"""
Linear regression and minimum squared code as explained by Dot CSV
https://www.youtube.com/watch?v=w2RJ1D6kz-o
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# Load dataset
boston = load_boston()
print(type(boston))

X = boston.data[:, 5]
Y = boston.target

plt.scatter(X, Y, alpha=.3)

# Add column of ones for constant values.
X = np.array([np.ones(506), X]).T

B = np.linalg.inv(X.T @ X) @ X.T @ Y
print(B)

plt.plot([4,9], [4 * B[1] + B[0], B[0] + B[1] * 9], c="red")
plt.show()
