"""
# Single neuron in tensorflow, Part 3: Logistic Regression
Created on Mon Sep 16 18:26:24 2019

@author: felip
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_openml

# Load and extract the dataset
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]

# Convert from str to float 
y = y.astype(np.float)

print(X.shape)
print(y.shape)

# Check the distribtuion of the digits
for i in range(10):
    print ("digit", i, "appear", np.count_nonzero(y == i), "times")
    
# Plot digit function
def plot_digit(some_digit):
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = plt.cm.binary, 
               interpolation = "nearest")
    plt.axis("off")
    plt.show()

# Sample digit
plot_digit(X[36003])    

# Extract the labels and the digits 1 and 2
X_train = X[np.any([y == 1, y == 2], axis = 0)]
y_train = y[np.any([y == 1, y == 2], axis = 0)]

print(X_train.shape)
print(y_train.shape)

# Normalize data
X_train_normalised = X_train / 255.0

X_train_tr = X_train_normalised.transpose()
y_train_tr = y_train.reshape(1, y_train.shape[0])
n_dim = X_train_tr.shape[0]
dim_train = X_train_tr.shape[1]

print(X_train_tr.shape)
print(y_train_tr.shape)
print(n_dim)
print("The training dataset has", dim_train, "observations (m).")

# Rescale the labels, zero labels are digit 1, one labels are digit 2
y_train_shifted = y_train_tr - 1

Xtrain = X_train_tr
ytrain = y_train_shifted

# Network architecture
tf.reset_default_graph()

# Input and output of the network
X = tf.placeholder(tf.float32, [n_dim, None])
Y = tf.placeholder(tf.float32, [1, None])
learning_rate = tf.placeholder(tf.float32, shape=())

# Weight and bias definition
W = tf.Variable(tf.zeros([1, n_dim]))
b = tf.Variable(tf.zeros(1))

init = tf.global_variables_initializer()

# Consists in the equation y = f(z) = w * X + b
y_ = tf.sigmoid(tf.matmul(W, X) + b)
cost = - tf.reduce_mean(Y * tf.log(y_) + (1 - Y) * tf.log(1 - y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Learning and running the model function
def run_logistic_model(learning_r, training_epochs, train_obs, train_labels, 
                       debug = False):
    sess = tf.compat.v1.Session()
    sess.run(init)
    
    cost_history = np.empty(shape=[0], dtype = float)
    
    for epoch in range(training_epochs + 1):
        #print('epoch: ', epoch)
        sess.run(b, feed_dict = {X: train_obs, Y: train_labels,
                                 learning_rate: learning_r})
        sess.run(training_step, feed_dict = {X: train_obs, Y: train_labels,
                                             learning_rate: learning_r})
        sess.run(b, feed_dict = {X: train_obs, Y: train_labels, 
                                 learning_rate: learning_r})
        cost_ = sess.run(cost, feed_dict = {X: train_obs, Y: train_labels, 
                                            learning_rate: learning_r})
        cost_history = np.append(cost_history, cost_)
        
        if (epoch % 500 == 0) & debug:
            print("Reached epoch ", epoch, " cost J = ", str.format('{0:.6f}',
            cost_))
        
    return sess, cost_history

# Model function
sess, cost_history = run_logistic_model(learning_r = 0.005, 
                                        training_epochs = 500,
                                        train_obs = Xtrain,
                                        train_labels = ytrain,
                                        debug = True)

# Obtain the accuracy of the model
correct_prediction = tf.equal(tf.greater(y_, 0.5), tf.equal(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict = {X: Xtrain, Y: ytrain, 
                                 learning_rate: 0.05}))

# Plot cost function vs epochs
plt.rc('font', family='arial')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
    
plt.tight_layout()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(cost_history, ls='solid', color = 'black', label = '$\gamma = 0.001$')
ax.set_xlabel('epochs', fontsize = 16)
ax.set_ylabel('Cost function $J$', fontsize = 16)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 16)
plt.tick_params(labelsize=16)
