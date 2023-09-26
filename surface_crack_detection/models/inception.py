import sys
sys.path.append('surface_crack_detection')
from utils.utils import split_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf

# load dataset
dataset = pd.read_csv('dataset/dataset_final.csv')

# split dataset
train_df, test_df = train_test_split(dataset, test_size=0.80, random_state=42)

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
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(base_model.input, x)

# using this class at the end of an epoch
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.999):
            self.model.stop_training = True


# compiling the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy',
              metrics=['accuracy'])

# showing model's summary
model.summary()


# training the model
callbacks = myCallback()
history = model.fit(train_data, validation_data=valid_data,
                    epochs=7, verbose=1, callbacks=[callbacks])

model.save('../models_file/inception_model.h5')
