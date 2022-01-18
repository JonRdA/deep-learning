"""Code of tutorial video by Dot CSV 
https://www.youtube.com/watch?v=W8AeOXa_FqU
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

# Create dataset
n = 500     # num samples
p = 2       # num classes

X, Y = make_circles(n_samples=n, factor=.5, noise=.05)

#plt.scatter(X[Y==0, 0], X[Y==0, 1], c="skyblue")
#plt.scatter(X[Y==1, 0], X[Y==1, 1], c="salmon")
#plt.axis("equal")
#plt.show()


class Layer:
    """Neural network layer class"""
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur) * 2 - 1
        self.W = np.random.rand(n_conn, n_neur) * 2 -1 


# Activation function sigmoid, tuple, value or derivative.
sigm = (lambda x: 1 / (1 + np.exp(-x)),
        lambda x: x * (1 - x))

def create_nn(topology, act_f):
    """Create neural network by creating Layer objects."""
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(Layer(topology[l], topology[l + 1], act_f))
    return nn

# Cost function
l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: (Yp - Yr))

def train(neural_net, X, Y, l2_cost, lr=.5, train=True):

    out = [(0, X)]
    
    # Forward pass
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b
        a = neural_net[l].act_f[0](z)
        out.append((z, a))

    if train:

        # Backward pass
        deltas = []

        for l in reversed(rante(len(neural_net))):
            if l == len(neural_net) - 1:
                # We are on last layer

            else:
                # Calculate delta with respect to previous layer

        # Gradient descent


topology = [p, 4, 8, 16, 8, 4, 1]
nn = create_nn(topology, sigm)
train(nn, X, Y, l2_cost, .5)
