
# -*- coding: utf-8 -*-
"""
# Feedforward Neural Networks Part 2: Zalando dataset
# Batch vs Mini-batch, Gradiend Descent training
Created on Wed Sep 25 19:04:14 2019

@author: felip
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import random

# Load csv file
# Download it from: https://www.kaggle.com/zalando-research/fashionmnist
data_train = pd.read_csv('fashionmnist/fashion-mnist_train.csv', header = 0)
data_test = pd.read_csv('fashionmnist/fashion-mnist_test.csv', header = 0)


# Train dataset preparation
labels = data_train['label'].values.reshape(1, 60000)

# Check clothes distribution
for i in range(10):
    print ("Item of clothing ", i, "appears ", np.count_nonzero(labels == i), "times")

# One-hot encoding, from scalar value to vector of (1, 10)
labels_ = np.zeros((60000, 10))
labels_[np.arange(60000), labels] = 1
labels_ = labels_.transpose()

train = data_train.drop('label', axis = 1).transpose()


# Test dataset preparation
labels_test = data_test['label'].values.reshape(1, 10000)

# One-hot encoding, from scalar value to vector of (1, 10)
labels_test_ = np.zeros((10000, 10))
labels_test_[np.arange(10000), labels_test] = 1
labels_test_ = labels_test_.transpose()

test = data_test.drop('label', axis=1).transpose()


# Normalize dataset
train = np.array(train / 255.0)
test = np.array(test / 255.0)
labels_ = np.array(labels_)
labels_test_ = np.array(labels_test_)


# Plot random cloth 
idx = random.randint(train.shape[0], train.shape[1])
fig = plt.figure(figsize=(12, 10))
plt.imshow(train[:,idx].reshape(28,28), cmap = matplotlib.cm.binary, interpolation = "nearest")
plt.axis("on")
plt.title("Random item of clothing")
plt.show()


# Tensorflow model implementation
n_dim = 784
tf.reset_default_graph()


# Number of neurons in the layers
n1 = 15  # Number of neurons in layer 1
n2 = 10 # Number of neurons in output layer

cost_history = np.empty(shape = [1], dtype = float)
learning_rate = tf.placeholder(tf.float32, shape = ())


# Network parameters
X = tf.placeholder(tf.float32, [n_dim, None])
Y = tf.placeholder(tf.float32, [10, None])
W1 = tf.Variable(tf.truncated_normal([n1, n_dim], stddev = 0.1))
b1 = tf.Variable(tf.constant(0.1, shape = [n1,1]) )
W2 = tf.Variable(tf.truncated_normal([n2, n1], stddev = 0.1))
b2 = tf.Variable(tf.constant(0.1, shape = [n2,1]) )


# Activation functions and mathematical operations
Z1 = tf.nn.relu(tf.matmul(W1, X) + b1)
Z2 = tf.nn.relu(tf.matmul(W2, Z1) + b2)
y_ = tf.nn.softmax(Z2, 0)


# Cost function and optimizer
cost = - tf.reduce_mean(Y * tf.log(y_) + (1 - Y) * tf.log(1 - y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()


# Training initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# Hyperparameter: number of iterations
training_epochs = 50


# Training the model with batch gradient descent (bad solution for learning)
print("Batch gradient descent training")
cost_history = []
for epoch in range(training_epochs + 1):
    sess.run(optimizer, feed_dict = {X: train, Y: labels_, learning_rate: 
        0.001})
    cost_ = sess.run(cost, feed_dict = {X: train, Y: labels_, learning_rate: 
        0.001})
    cost_history = np.append(cost_history, cost_)
    
    if (epoch % 10 == 0):
        print("Reached epoch ", epoch, "cost J = ", cost_)
    

# Accuracy calculation of the training set
correct_predictions = tf.equal(tf.argmax(y_, 0), tf.argmax(Y, 0))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
print("Accuracy: ", accuracy.eval({X: train, Y: labels_, learning_rate: 
        0.001}, session = sess))
    

# Training the model with mini-batch gradient descent
def model(miniBatch_size, training_epochs, features, classes, logging_step,
          learning_r):
    
    # Training initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    cost_history = []
    
    # Mini-batch training
    for epoch in range(training_epochs + 1):
        for i in range(0, features.shape[1], miniBatch_size):
            X_train_mini = features[:, i:i + miniBatch_size]
            Y_train_mini = classes[:, i:i + miniBatch_size]
            
            sess.run(optimizer, feed_dict = {X: X_train_mini,
                                             Y: Y_train_mini,
                                             learning_rate: learning_r})
        
        cost_ = sess.run(cost, feed_dict = {X: features,
                                             Y: classes,
                                             learning_rate: learning_r})
            
        cost_history = np.append(cost_history, cost_)
        
        if(epoch % logging_step == 0):
            print("Reached epoch ", epoch, " cost J = ", cost_)
    
    
    return sess, cost_history


# Training function
print("Mini-batch gradient descent training")
sess, cost_history1 = model(50, training_epochs, train, labels_, 10, 0.01)


# Predict accuracy
correct_predictions = tf.equal(tf.argmax(y_,0), tf.argmax(Y,0))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
print ("Accuracy:", accuracy.eval({X: train, Y: labels_, learning_rate: 0.001},
                                  session = sess))
    


