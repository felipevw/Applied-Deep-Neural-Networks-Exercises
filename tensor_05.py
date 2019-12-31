# -*- coding: utf-8 -*-
"""
Introduction to Regularization

Created on Tue Dec 31 15:55:10 2019

@author: felip
"""

import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

from sklearn.datasets import load_boston



# Import datasets
boston = load_boston()
features = np.array(boston.data)
target = np.array(boston.target)



# Normalize data
def normalize(dataset):
    mu = np.mean(dataset, axis = 0)
    sigma = np.std(dataset, axis = 0)
    return (dataset - mu) / sigma



# Split into training and dev datasets
features_norm = normalize(features)  
np.random.seed(42) # Fixes random numbers
rnd = np.random.rand(len(features_norm)) < 0.8

train_x = np.transpose(features_norm[rnd])
train_y = np.transpose(target[rnd])
dev_x = np.transpose(features_norm[~rnd])
dev_y = np.transpose(target[~rnd])



# Reshape the np vectors
train_y = train_y.reshape(1, len(train_y))
dev_y = dev_y.reshape(1, len(dev_y))



# Neural network architecture with 4 layers and 20 neurons
# Note weights and bias are returned
def create_layer(X, n, activation):
    ndim = int(X.shape[0])
    stddev = 2.0 / np.sqrt(ndim)
    initialization = tf.truncated_normal((n, ndim), stddev = stddev)
    W = tf.Variable(initialization)
    b = tf.Variable(tf.zeros([n ,1]))
    Z = tf.matmul(W, X) + b
    return activation(Z), W, b

tf.reset_default_graph()



# Network definition
n_dim = 13
n1 = 20
n2 = 20
n3 = 20
n4 = 20
n_outputs = 1

tf.set_random_seed(5)

X = tf.placeholder(tf.float32, [n_dim, None])
Y = tf.placeholder(tf.float32, [1, None])

learning_rate = tf.placeholder(tf.float32, shape=())

hidden1, W1, b1 = create_layer(X, n1, activation = tf.nn.relu)
hidden2, W2, b2 = create_layer(hidden1, n2, activation = tf.nn.relu)
hidden3, W3, b3 = create_layer(hidden2, n3, activation = tf.nn.relu)
hidden4, W4, b4 = create_layer(hidden3, n4, activation = tf.nn.relu)
y_, W5, b5 = create_layer(hidden4, n_outputs, activation = tf.identity)

cost = tf.reduce_mean(tf.square(y_ - Y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.9,
                                   beta2 = 0.999, epsilon = 1e-8).minimize(cost)



# Network training, hardcoded
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_train_history = []
cost_dev_history = []

for epoch in range(10000 + 1):
    sess.run(optimizer, feed_dict = {X: train_x, Y: train_y, learning_rate: 0.001})
    cost_train_ = sess.run(cost, feed_dict={X:train_x, Y: train_y, 
                                            learning_rate: 0.001})
    cost_dev_ = sess.run(cost, feed_dict={X:dev_x, Y: dev_y, 
                                            learning_rate: 0.001})
    cost_train_history = np.append(cost_train_history, cost_train_)
    cost_dev_history = np.append(cost_dev_history, cost_dev_)

    if (epoch % 1000 == 0):
        print("Reached epoch", epoch, "cost J(train) = ", cost_train_)
        print("Reached epoch", epoch, "cost J(dev) = ", cost_dev_)



# Plot resulting costs
plt.plot(cost_train_history, label = 'MSE in training', color = 'b')
plt.plot(cost_dev_history, label = 'MSE in testing', color = 'r')
plt.legend()