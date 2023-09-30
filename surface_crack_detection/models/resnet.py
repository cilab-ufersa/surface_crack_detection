import sys
sys.path.append('surface_crack_detection')
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.utils import split_data

# load dataset
dataset = pd.read_csv('dataset/dataset_final.csv')

# split dataset
train_df, test_df = train_test_split(dataset.sample(
    frac=1.0, random_state=42), train_size=0.80, random_state=42)

preprocess_input = tf.keras.applications.resnet.preprocess_input

# train, validation and test data
train_data, valid_data, test_data = split_data(
    train_df, test_df, image_channels=1.0, preprocess_input=preprocess_input)

# load the path to weights file
load_weights_file = 'surface_crack_detection/models/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# using the ResNet50 architecture
base_model = tf.keras.applications.resnet50.ResNet50(
    include_top=False, weights=load_weights_file, input_shape=(227, 277, 3), pooling='avg')

for layer in base_model.layers:
    layer.trainable = False

# building the model
x = tf.keras.layers.Flatten(base_model.output)
x = tf.keras.layers.Dense(512, 'relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, 'relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, 'relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, 'softmax')(x)

model = tf.keras.Model(base_model.input, x)

# compiling the modl
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy'])

# showing model's summary
model.summary()

# training the model
history = model.fit(train_data, validation_data=valid_data,
                    epochs=10, verbose=1)
