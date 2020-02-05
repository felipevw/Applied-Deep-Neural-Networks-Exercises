# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:25:03 2020

@author: felip
"""

import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")


import datetime
import time

NAME = "MNIST-{}".format(int(time.time()))

#log_dir1 = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = TensorBoard(log_dir = 'logs\{}'.format(NAME))

# Load fashion mnist
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize data
train_images, test_images = train_images / 255.0, test_images / 255.0



# Define model
def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])




# Define training procedure
model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



#log_dir="logs/fit/" 


#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir1, histogram_freq=1)




model.fit(x=train_images, 
          y=train_labels, 
          epochs=2, 
          validation_data=(test_images, test_labels), 
          callbacks=[tensorboard])




