# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:17:03 2020

@author: felip
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras



# Tensorboard setup
# Directory for the tensorboard
import os
root_logdir = os.path.join(os.curdir, "my_logs")



def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)



run_logdir = get_run_logdir()




# Load the dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()



# Explore the dataset
W_grid = 4
L_grid = 4

plt.close()
fig, axes = plt.subplots(L_grid, W_grid, figsize = (15, 15))
axes = axes.ravel()

n_training = len(X_train)

for i in np.arange(0, L_grid * W_grid):
    index = np.random.randint(0, n_training)
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off')

plt.subplots_adjust(hspace = 0.5)



# Data preparation
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')




# One hot encoding
num_cat = len(np.unique(y_train))

y_train = keras.utils.to_categorical(y_train, num_cat)
y_test = keras.utils.to_categorical(y_test, num_cat)



# Normalize all values
X_train = X_train / 255
X_test = X_test / 255



# Input shape for the CNN
input_shape = X_train.shape[1:]

# Activation function swish
def swish(x, beta = 1):
    return (x * tf.keras.backend.sigmoid(beta * x))


keras.utils.get_custom_objects().update(
    {"swish": keras.layers.Activation(swish)})



# CNN architecture with sequential API
acti = "swish"
model = tf.keras.models.Sequential([

    keras.layers.Conv2D(32, 3, activation = acti, input_shape = input_shape),
    keras.layers.Conv2D(32, 3, activation = acti),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.3),

    keras.layers.Conv2D(64, 3, activation = acti),
    keras.layers.Conv2D(64, 3, activation = acti),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.3),

    keras.layers.Flatten(),

    keras.layers.Dense(1024, activation = acti),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(1024, activation = acti),

    keras.layers.Dense(10, activation = 'softmax'),

])

model.summary()



# Training the neural network
nadam = keras.optimizers.Nadam(learning_rate=0.001,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07)

model.compile(optimizer = nadam, loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, batch_size = 512, epochs = 2,  
                    callbacks = [tensorboard_cb])



# Model evaluation with confusion matrix
evaluation = model.evaluate(X_test, y_test)
print('Test accuracy: {}'.format(evaluation[1]))

"""
Results:
Train: 68.52% accuarcy, test: 69.76% accuracy in 10 epochs

"""













