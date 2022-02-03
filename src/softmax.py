"""Multiclass logistic regression classifier (softmax) implementation."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class LogReg:
    """Multiclass logistic regression classifier

    Attributes:
        X (tensor): input data.
        Y (tensor): input data class.

        m (int): number of instances.
        n (int): number of variables.
        c (int): number of classes.
        """

    def __init__(self, X, Y):
        """Initialize logistic regression and parameter settings.

        Args:
            X (tensor): input data.
            Y (tensor): class data.
        """
        self.X, self.Y = self.preprocess_data(X, Y)
        
        self.m, self.n = self.X.shape       # instances, variables
        self.c = self.Y.shape[1]            # classes

        # Training hyperparameters
        self.ep = 500       # epochs
        self.bs = 256       # minibatch size
        self.lr = .1        # learning rate
        self.df = 50        # display frequency, epochs
    
    def preprocess_data(self, X, Y):
        """Add ones column to X data and one hot encode Y.

        Args:
            X (tensor): input data.
            Y (tensor): class data.

        Returns:
            Xp, Yp (tensors): preprocessed data.
        """
        Xp = tf.concat([tf.ones((X.shape[0], 1)), X], axis=1)
        (cs, *_) = tf.unique_with_counts(Y)
        c = cs.shape[0]
        Yp = tf.one_hot(Y, depth=c)
        return Xp, Yp

    def predict(self, W, X):
        """Make prediction of values for logistic regression, softmax.

        Args:
            W (tensor): parameters.
            X (tensor): input data.

        Returns:
            h0 (tensor): prediction of logistic regression probabilities.
        """
        z = tf.matmul(X, W)
        h0 = tf.nn.softmax(z)
        return tf.clip_by_value(h0, 1e-9, 1.)       # clip to avoid log(0) err

    def loss(self, h0, Y):
        """Compute loss of prediction with corss entropy function.

        Args:
            h0 (tensor): y value predictions probabilities.
            Y (tensor): true values of class, one hot encoded.

        Returns:
            ls (tensor): loss function value per instance.
        """
        sel = Y * tf.math.log(h0)       # selected class to compute error
        ls = - tf.reduce_sum(sel, axis=1)
        return ls

    def gradient(self, W, X, Y):
        """Compute gradient of loss function for location W.
        
        Args:
            W (tensor): parameters.
            X (tensor): input data.

        Returns:
            grad (tensor): partial derivatives of loss function for parameters.
        """
        h0 = self.predict(W, X)
        err = Y - h0
        grad = - tf.matmul(tf.transpose(X), err)
        return grad

    def shuffle_data(self, indx):
        """Shuffle the X & Y of the data.

        Args:
            X (tensor): input data.
            Y (tensor): class data.
            indx (tensor): index array shuffle and select data.

        Returns:
            Xs, Yx (tensors): shuffled data.
        """
        ind = tf.random.shuffle(indx)
        Xs = tf.gather(self.X, indices=ind)
        Ys = tf.gather(self.Y, indices=ind)
        return Xs, Ys

    def get_minibatches(self, X, Y):
        """Batch generator from shuffled data.

        Args:
            X (tensor): input data.
            Y (tensor): class data.

        Yields:
            Xb, Yb (tensors): minibatch tensors.
        """
        count = 0
        m = X.shape[0]
        i_0, i_1 = 0, self.bs       # selection indices
        while i_0 < m:
            count += 1
            Xb = X[i_0:i_1, :]
            Yb = Y[i_0:i_1, :]
            i_0 += self.bs
            i_1 += self.bs
            yield Xb, Yb

    def train(self, hyp=None):
        """Train model stochastic gradient descent. Optional tuple hyperparams.

        Args:
            hyp (tuple): hyperparameters (ep, bs, lr, df).

        Returns:
            W (tensor): updated parameters.
        """
        if hyp is not None:
            self.ep, self.bs, self.lr, self.df = hyp

        indx = tf.range(self.m)
        W = tf.random.uniform((self.n, self.c))

        for epoch in range(self.ep):
            Xs, Ys = self.shuffle_data(indx)

            grd = tf.zeros((self.n, self.c))
            for i, (Xb, Yb) in enumerate(self.get_minibatches(Xs, Ys)):
                grd += self.gradient(W, Xb, Yb)

            grd = grd / (i + 1)
            W -= self.lr * grd

            # TODO stats displaying to function
            if (epoch % self.df) == 0:
                h0 = self.predict(W, Xb)
                ls = self.loss(h0, Yb)
                tf.print("Epoch:", epoch, "Loss:", tf.reduce_sum(ls, axis=0))

        return W

    def display_stats(self, epoch, W, X):
        """Display & save training statistics."""
        pass
#        if epoch % self.df == 0:
#            h0 = predict(W, X)
#            ls = loss(h0, Y)
#            tf.print ("Epoch:", (epoch+1), "Cost:", avg_cost, \
#                "Dev acc:", evaluate_classifier(self.classify, dev_set[0:500]), \
#                "Train acc:", evaluate_classifier(self.classify, training_set[0:500]))

# TODO think about evaluation and classification.
#    def classify(self, W, X):
#        """Clasify database `X` by predicting instances classes
#
#        Args:
#
#            W (tensor): parameters.
#            X (tensor): data.
#
#        Returns:
#            Y_pred (tensor): predicted class, integers.
#        """
#        probs = self.predict(W, X)
#        Y_pred = tf.argmax(probs, axis=1)
#        return Y_pred
#
#def accuracy(Y, Y_pred):
#    probs = self.predict(W, X)
#    Y_pred = 
#    hypotheses = classifier(eval_set)
#    for i, example in enumerate(eval_set):
#        hypothesis = hypotheses[i]
#        if hypothesis == example['label']:
#            correct += 1        
#    return correct / float(len(eval_set)
#
