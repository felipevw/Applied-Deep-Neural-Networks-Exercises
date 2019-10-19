# -*- coding: utf-8 -*-
"""
Training Neural Networks Part 1: Self-Developed Optimizer

Created on Fri Oct 11 09:19:43 2019

@author: felip
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load data from tensorflow directly
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)



X, y = mnist["data"], mnist["target"]


# Convert from str to float 
y = y.astype(np.float)


# Extract the labels and the digits 1 and 2
X_train = X[np.any([y == 1, y == 2], axis = 0)]
y_train = y[np.any([y == 1, y == 2], axis = 0)]


# Normalize data
X_train_normalised = X_train / 255.0

X_train_tr = X_train_normalised.transpose()
y_train_tr = y_train.reshape(1, y_train.shape[0])
n_dim = X_train_tr.shape[0]
dim_train = X_train_tr.shape[1]


# Rescale the labels, zero labels are digit 1, one labels are digit 2
y_train_shifted = y_train_tr - 1
labels_ = y_train_shifted






# Network architecture
X = tf.placeholder(tf.float32, [784, None])     # mnist data image shape
Y = tf.placeholder(tf.float32, [10, None])      # 0-9 digit recognition
learning_rate = tf.placeholder(tf.float32, shape = ())
W = tf.Variable(tf.zeros([10, 784]), dtype = tf.float32)
b = tf.Variable(tf.zeros([10, 1]), dtype = tf.float32)

y_ = tf.nn.softmax(tf.matmul(W, X) + b)
cost = - tf.reduce_mean(Y * tf.log(y_) + (1 - Y) * tf.log(1 - y_))


# Compute stochastic gradient descent
grad_W, grad_b = tf.gradients(xs = [W, b], ys = cost)

new_W = W.assign(W - learning_rate * grad_W) 
new_b = b.assign(b - learning_rate * grad_b) 


# Training initialization function
def run_model_mb(minibatch_size, training_epochs, features, classes, 
                 logging_step = 100, learning_r = 0.001):
    sess = tf.Session();
    sess.run(tf.global_variables_initializer())
    
    total_batch = int(mnist.train.num_examples / minibatch_size)
    
    cost_history = []
    accuracy_history = []
     
    correct_prediction = tf.equal(tf.greater(y_, 0.5), tf.equal(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    for epoch in range(training_epochs + 1):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(minibatch_size)
            batch_xs_t = batch_xs.T
            batch_ys_t = batch_ys.T
            
            _, _, cost_ = sess.run([new_W, new_b, cost], feed_dict = {X: batch_xs_t,
                                   Y: batch_ys_t, learning_rate: learning_r})
    
        cost_ = sess.run(cost, feed_dict = {X: features, Y: classes})
        accuracy_ = sess.run(accuracy, feed_dict = {X: features, Y: classes})
        cost_history = np.append(cost_history, cost_)
        accuracy_history = np.append(accuracy_history, accuracy_)
        
        if (epoch % logging_step == 0):
            print("Reached epoch", epoch, "cost J =", cost_)
            print("Accuracy:", accuracy_)
        
    return sess, cost_history, accuracy_history

# Call the training function
sess, cost_history, accuracy_history = run_model_mb(100, 50, X_train_tr, 
                                                    labels_, logging_step = 10, 
                                                    learning_r = 0.01)
            





