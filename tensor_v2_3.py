# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:26:28 2020

@author: felip
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random 
from PIL import Image


# Load the imagenet weights
model = tf.keras.applications.ResNet50(weights = 'imagenet')



# Evaluate the pretrained model
sample_image= tf.keras.preprocessing.image.load_img(
    'D:\\Python\\tensorflow2\\bicycle.png', target_size = (224, 224))

#plt.close()
#plt.imshow(sample_image)



# Process the input image
sample_array = tf.keras.preprocessing.image.img_to_array(sample_image)
sample_image_exp = np.expand_dims(sample_array, axis = 0)
sample_process = tf.keras.applications.resnet50.preprocess_input(sample_image_exp)



# Predict the model
predictions = model.predict(sample_process)
print('predictions:', tf.keras.applications.resnet50.decode_predictions(
    predictions, top = 5)[0])
print('predictions:', tf.keras.applications.resnet50.decode_predictions(
    predictions, top = 1)[0])



# Load the ResNet50 architecture with the imagenet dataset weight
base_model = tf.keras.applications.ResNet50(weights = 'imagenet', 
                                            include_top = False)



#print(base_model.summary())


# Append to the end of the network a new architeture
new_model = tf.keras.Sequential([
    base_model, 
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(2, activation = 'softmax') 
    ])

#print(new_model.summary())


# Proprocess input funtion
preprocessing_function = tf.keras.applications.resnet50.preprocess_input
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function)



# Set path of the training data
train_generator = train_datagen.flow_from_directory(
    'D:\\Python\\tensorflow2\\Transfer Learning Data\\train', 
    target_size = (224, 224),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True)


# Network training
new_model.compile(optimizer = 'Nadam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

history = new_model.fit_generator(
    generator = train_generator, 
    steps_per_epoch = train_generator.n//train_generator.batch_size, 
    epochs = 5)















