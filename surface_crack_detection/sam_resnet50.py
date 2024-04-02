import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from sklearn.model_selection import train_test_split
from utils.utils import split_data


dataset = pd.read_csv('') # Path to the dataset

train_df, test_df = train_test_split(dataset.sample(frac=1.0, random_state=42), test_size=0.2, random_state=42)

preprocess_input = tf.keras.applications.resnet.preprocess_input

train_data, valid_data, test_data = split_data(
    train_df, test_df, image_channels=1, preprocess_input=preprocess_input, image_width=150, image_height=150
)


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.999:
            print('\nReached 99.9% accuracy sp cancelling training')
            self.model.stop_training = True


callbacks = MyCallback()

pre_trained_model = tf.keras.applications.resnet.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(150, 150, 3),
    pooling='avg'
)

for layer in pre_trained_model.layers:
    layer.trainable = False

x = pre_trained_model.output
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(5, activation='softmax')(x)

model = tf.keras.Model(pre_trained_model.input, output)

model.summary()

steps_per_epoch_training = len(train_data)
steps_per_epoch_validation = len(valid_data)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=30,
    verbose=1,
    steps_per_epoch=steps_per_epoch_training,
    validation_steps=steps_per_epoch_validation,
    callbacks=[callbacks]
)

history = history.history

with open('/content/drive/MyDrive/sam_resnet50/history/sam_resnet50_history.h5', 'wb') as f:
    pickle.dump(history, f)

model.save('/content/drive/MyDrive/sam_resnet50/models/sam_resnet50.h5')
