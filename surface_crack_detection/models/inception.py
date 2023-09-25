import sys
sys.path.append('surface_crack_detection')
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.utils import split_data

# load dataset
dataset = pd.read_csv('dataset/dataset_final.csv')

# split dataset
train_df, test_df = train_test_split(dataset.sample(
    6000, random_state=42), test_size=0.80, random_state=42)

# train, validation and test data
train_data, valid_data, test_data = split_data(train_df, test_df)

# load the path to weights file
load_weights_file = 'surface_crack_detection/models/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# using the inceptionV3 architecture
base_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False, input_shape=(227, 227, 3), weights=load_weights_file)

# making all the layers in base_model non-trainable
for layer in base_model.layers:
    layer.trainable = False

# using the part of base_model from input later until the layer 'mixed7'
last_layer = base_model.get_layer('mixed7')
last_output = last_layer.output

# building the model
model = tf.keras.layers.Flatten()(last_output)
model = tf.keras.layers.Dense(1024, activation='relu')(model)
model = tf.keras.layers.Dropout(0.2)(model)
model = tf.keras.layers.Dense(1, 'sigmoid')(model)
model = tf.keras.Model(base_model.input, model)

# compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# showing model's summary
model.summary()

# training the model
history = model.fit(train_data, validation_data=valid_data,
                    epochs=50, verbose=1)

model.save('../models_file/inception_model.h5')
