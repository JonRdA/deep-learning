"""Deep learning main module for calling neural network and regressions."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from softmax import LogReg

def test():
    """Main testing and debugging code."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

    # Flatten images to 1-D vector of 784 features (28*28).
    x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])

    # Normalize images value from [0, 255] to [0, 1].
    x_train, x_test = x_train / 255., x_test / 255

    m = 200     # instances
    X = tf.constant(x_train[:m])
    Y = tf.constant(y_train[:m])

    mm = 800
    ep = 200       # epochs
    bs = 256       # batch size
    lr = .01       # learning rate
    df = 50        # display frequency

    a = LogReg(x_train[:mm], y_train[:mm])
    W = a.train((ep, bs, lr, df))

def main():
    """Main code to run directly."""
    test()

if __name__ == "__main__":
    main()

