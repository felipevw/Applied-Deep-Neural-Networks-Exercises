# -*- coding: utf-8 -*-
"""
Feedforward Neural Networks Part 3: Zalando dataset improved

Created on Tue Oct  1 17:05:24 2019

@author: felip
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt


# Load csv file
# Download it from: https://www.kaggle.com/zalando-research/fashionmnist
data_train = pd.read_csv('fashionmnist/fashion-mnist_train.csv', header = 0)
data_test = pd.read_csv('fashionmnist/fashion-mnist_test.csv', header = 0)


# Get the labels of each image
labels = data_train['label'].values.reshape(1, 60000)


# One-hot encoding, from scalar value to vector of (1, 10)
labels_ = np.zeros((60000, 10))
labels_[np.arange(60000), labels] = 1
labels_ = labels_.transpose()

# Get the training images
train = data_train.drop('label', axis = 1).transpose()


# Get the labels of each image for testing dataset
labels_test = data_test['label'].values.reshape(1, 10000)


# One-hot encoding, from scalar value to vector of (1, 10) for testing dataset
labels_test_ = np.zeros((10000, 10))
labels_test_[np.arange(10000), labels_test] = 1
labels_test_ = labels_test_.transpose()


# Get the testing images
test = data_test.drop('label', axis=1).transpose()


# Normalize training and testing datasets
train = np.array(train / 255.0)
test = np.array(test / 255.0)
labels_ = np.array(labels_)
labels_test_ = np.array(labels_test_)


# Layer creation function
# X is the input of the layer
# n are the neurons of it
# activation is the tensorflow activation function
def create_layer (X, n, activation):
    
    # Size of the input of the layer
    ndim = int(X.shape[0])
    
    # Standard deviation, He initialization strategy
    stddev = 2 / np.sqrt(ndim)
    
    # Weight initialization
    init = tf.truncated_normal((n, ndim), stddev = stddev)
    
    # Layer structure
    W = tf.Variable(init)
    b = tf.Variable(tf.zeros([n, 1]))
    Z = tf.matmul(W, X) + b
    
    # Return layer with activation function
    return activation(Z)


# Network parameters definition
n_dim = 784 
n1 = 10
n2 = 10
n_outputs = 10

X = tf.placeholder(tf.float32, [n_dim, None])
Y = tf.placeholder(tf.float32, [n_outputs, None])

learning_rate = tf.placeholder(tf.float32, shape = ())

hidden1 = create_layer(X, n1, activation = tf.nn.relu)
hidden2 = create_layer(hidden1, n2, activation = tf.nn.relu)
outputs = create_layer(hidden2, n_outputs,  activation = tf.identity)

y_ = tf.nn.softmax(outputs)


# Cost and optimization functions
cost = - tf.reduce_mean(Y * tf.log(y_) + (1 - Y) * tf.log(1 - y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()


# Training initialization function
def training_model(minibatch_size, training_epochs, features, classes, 
                   logging_step = 10, learning_r = 0.001):
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    cost_history = []
    for epoch in range(training_epochs + 1):
        for i in range(0, features.shape[1], minibatch_size):
            X_train_mini = features[:, i:i + minibatch_size]
            Y_train_mini = classes[:, i:i + minibatch_size]
            
            sess.run(optimizer, feed_dict = {X: X_train_mini, Y: Y_train_mini,
                                             learning_rate: learning_r})
    
        cost_ = sess.run(cost, feed_dict = {X: features, Y: classes,
                                            learning_rate: learning_r})
    
        cost_history = np.append(cost_history, cost_)
            
        if (epoch % logging_step == 0):
            print("Reached epoch", epoch, "cost J = ", cost_)
                
    return sess, cost_history


# Run the training function
sess, cost_history = training_model (25, 100, train, labels_, logging_step = 10, 
                            learning_r = 0.01)
    

# Predict accuracy
correct_predictions = tf.equal(tf.argmax(y_,0), tf.argmax(Y,0))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
print ("Accuracy:", accuracy.eval({X: train, Y: labels_, learning_rate: 0.001},
                                  session = sess))
