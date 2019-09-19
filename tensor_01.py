
"""
# Computational graphs in tensorflow
Created on Mon Sep 16 18:26:24 2019

@author: felip
"""
# Testing simple computational graphs in tensorflow


import sys

import tensorflow.keras
import tensorflow as tf

print(f"Python {sys.version}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")

# Graph 1: Sum of two tensors x1 + x2

# Session 
with tf.compat.v1.Session() as sess:

    # With tf.constant variable
    x1 = tf.constant(1)
    x2 = tf.constant(2)
    z = tf.add(x1, x2)

    # Call Session and check result
    print("Tensor result is ", sess.run(z))


    # With tf.Variable
    x1 = tf.Variable(1)
    x2 = tf.Variable(2)
    z = tf.subtract(x1, x2)
    
    # Call Session and check result
    init = tf.global_variables_initializer()
    sess.run(init)
    print("Tensor result is ", sess.run(z))


    # With tf.placeholder, redifined to use arrays with two elements
    x1 = tf.placeholder(tf.float32, [2])
    x2 = tf.placeholder(tf.float32, [2])
    z = tf.add(x1, x2)
    
    # Assign values with dictionary
    feed_dict = { x1: [1, 5], x2: [1, 1] }
    
    # Call Session and check result
    print("Tensor result is ", sess.run(z, feed_dict))
    
    
    # Graph 2: calculating (x1 * w1) + (x2 * w2)
    x1 = tf.placeholder(tf.float32, 1)
    w1 = tf.placeholder(tf.float32, 1)
    x2 = tf.placeholder(tf.float32, 1)
    w2 = tf.placeholder(tf.float32, 1)
    
    z1 = tf.multiply(x1, w1)
    z2 = tf.multiply(x2, w2)
    z3 = tf.add(z1, z2)
    
    # Assign values with dictionary
    feed_dict = {x1: [1], w1: [2], x2: [3], w2: [4]}
    
    # Call Session and check result
    print("Tensor result is ", sess.run(z3, feed_dict))

sess.close()        # Free resources
