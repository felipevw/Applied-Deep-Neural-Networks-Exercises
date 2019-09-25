"""
# Single neuron in tensorflow, Part 2: Linear Regression
Created on Mon Sep 16 18:26:24 2019

@author: felip
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston

# Load dataset and extract features and labels
boston = load_boston()
features = np.array(boston.data)
labels = np.array(boston.target)

print(boston["DESCR"])

# Check observations
n_training_samples = features.shape[0]
n_dim = features.shape[1]

print('The dataset has ', n_training_samples, 'training samples.')
print('The dataset has ', n_dim, 'features.')

# Normalize function
def normalize(dataset):
    mu = np.mean(dataset, axis = 0)
    sigma = np.std(dataset, axis = 0)
    return (dataset - mu) / sigma

# Normalize dataset
features_norm = normalize(features)

# Transpose matrices
train_x = np.transpose(features_norm)
train_y = np.transpose(labels)

print(train_x.shape)
print(train_y.shape)

# Split the dataset into training and testing
np.random.seed(42)
rnd = np.random.rand(len(features_norm)) < 0.8

train_x = np.transpose(features_norm[rnd])
train_y = np.transpose(labels[rnd])
test_x = np.transpose(features_norm[~rnd])
test_y = np.transpose(labels[~rnd])

print(train_x.shape)
print(train_y.shape)

# Reshape vectors for tensorflow input
train_y = train_y.reshape(1, len(train_y))
test_y = test_y.reshape(1,len(test_y))
print(train_y.shape)
print(test_y.shape)

# Neuron and cost function definition
tf.compat.v1.reset_default_graph()

# Single neuron architecture
# Consists in the equation y = f(z) = w * X + b
X = tf.placeholder(tf.float32, [n_dim, None])           # matrix X
Y = tf.placeholder(tf.float32, [1, None])               # matrix y
learning_rate = tf.placeholder(tf.float32, shape=())    # learning rate parameter
W = tf.Variable(tf.ones([n_dim, 1]))                    # matrix of weights
b = tf.Variable(tf.zeros(1))                            # matrix of biases

# Cost and training parameters, note that the identity activation function is used
init = tf.global_variables_initializer()
y_ = tf.matmul(tf.transpose(W), X) + b                  # Output of the neuron
cost = tf.reduce_mean(tf.square(y_ - Y))                # Cost function

# Cost function optimizer
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# Learning and running the model function
def run_linear_model(learning_r, training_epochs, train_obs, train_labels,
                     debug = False):
    sess = tf.compat.v1.Session()
    sess.run(init)
    
    cost_history = np.empty(shape=[0], dtype = float)
    
    for epoch in range(training_epochs + 1):
        sess.run(training_step, feed_dict = {X: train_obs, Y: train_labels, 
                                              learning_rate: learning_r})
        cost_ = sess.run(cost, feed_dict = {X: train_obs, Y: train_labels, 
                                            learning_rate: learning_r})
        cost_history = np.append(cost_history, cost_)
        
        if (epoch % 1000 == 0) & debug:
            print("Reached epoch ", epoch, " cost J = ", str.format('{0:.6f}',
            cost_))
        
    return sess, cost_history


# Initialize the 
sess, cost_history = run_linear_model(learning_r = 0.01, 
                                     training_epochs = 10000,
                                     train_obs = train_x,
                                     train_labels = train_y,
                                     debug = True)


# Plot the cost function
plt.rc('font', family='arial')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

plt.tight_layout()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1, 1, 1)
ax.plot(cost_history, ls='solid', color = 'black')
ax.set_xlabel('epochs', fontsize = 16)
ax.set_ylabel('Cost function $J$ (MSE)', fontsize = 16)
plt.xlim(0,200)
plt.tick_params(labelsize=16)


# Predicted vs measured
pred_y = sess.run(y_, feed_dict = {X: test_x, Y: test_y})
mse = tf.reduce_mean(tf.square(pred_y - test_y))

plt.rc('font', family='arial')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
    
plt.tight_layout()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(test_y, pred_y, lw = 5)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw = 5)
ax.set_xlabel('Measured Target Value', fontsize = 16)
ax.set_ylabel('Predicted Target Value', fontsize = 16)
plt.tick_params(labelsize=16)



