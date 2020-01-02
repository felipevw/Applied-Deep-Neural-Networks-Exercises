# -*- coding: utf-8 -*-
"""
L2 Regularization Method Implementation

Created on Thu Jan  2 19:48:43 2020

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


# Regularization method L2
lambd = tf.placeholder(tf.float32, shape = ())
reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W1) + tf.nn.l2_loss(W1) + \
tf.nn.l2_loss(W1) + tf.nn.l2_loss(W1)

cost_mse = tf.reduce_mean(tf.square(y_ - Y))
cost = tf.reduce_mean(cost_mse + lambd * reg)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.9,
                                   beta2 = 0.999, epsilon = 1e-8).minimize(cost)



# Network training function
def model(training_epochs, features, target, logging_step = 100, learning_r = 0.001, 
    lambd_val = 0.1):

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    cost_history = []
    for epoch in range(training_epochs + 1):
        
        sess.run(optimizer, feed_dict = {X: features, Y: target, learning_rate:
            learning_r, lambd: lambd_val})
        cost_ = sess.run(cost_mse, feed_dict = {X: features, Y: target, learning_rate:
            learning_r, lambd: lambd_val})
        cost_history = np.append(cost_history, cost_)
        
        
        
        if(epoch % logging_step == 0):
            pred_y_test = sess.run(y_, feed_dict = {X: dev_x, Y: dev_y})
            
            print("Reached epoch ", epoch, " cost J = ", cost_)
            print("Training MSE = ", cost_)
            print("Dev MSE      = ", sess.run(cost_mse, feed_dict = {X: 
                dev_x, Y: dev_y}))
    
    return sess, cost_history



# Training definition
sess, cost_history = model(learning_r = 0.01, 
                           training_epochs = 5000,
                           features = train_x,
                           target = train_y,
                           logging_step = 5000,
                           lambd_val = 9.0)



# Plot weight distribution, best fit for lambda is 9.5 but for studying 
# the distribution it will be left at 0.0
weights1 = sess.run(W1,  feed_dict = {X: train_x, Y: train_y, learning_rate: 0.01, lambd: 0.0})
weights2 = sess.run(W2,  feed_dict = {X: train_x, Y: train_y, learning_rate: 0.01, lambd: 0.0})
weights3 = sess.run(W3,  feed_dict = {X: train_x, Y: train_y, learning_rate: 0.01, lambd: 0.0})
weights4 = sess.run(W4,  feed_dict = {X: train_x, Y: train_y, learning_rate: 0.01, lambd: 0.0})
weights5 = sess.run(W5,  feed_dict = {X: train_x, Y: train_y, learning_rate: 0.01, lambd: 0.0})



fig = plt.figure(figsize=(12, 8))
plt.tight_layout()



ax = fig.add_subplot(2, 2, 1)
plt.hist(weights1.flatten(), bins = 10)
ax.set_xlabel('Weights Layer 1', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
plt.ylim(0,90)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 


ax = fig.add_subplot(2, 2, 2)
plt.hist(weights2.flatten(), bins = 10)
ax.set_xlabel('Weights Layer 2', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
plt.ylim(0,90)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 


ax = fig.add_subplot(2, 2, 3)
plt.hist(weights3.flatten(), bins = 10)
ax.set_xlabel('Weights Layer 3', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
plt.ylim(0,180)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 


ax = fig.add_subplot(2, 2, 4)
plt.hist(weights4.flatten(), bins = 10)
ax.set_xlabel('Weights Layer 4', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
plt.ylim(0,180)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 



