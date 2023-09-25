# necessary libraries
import sys
sys.path.append('surface_crack_detection')
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils.utils import split_data
import pickle

#load dataset
dataset = pd.read_csv('dataset/dataset_final.csv')

#split dataset
train_df, test_df = train_test_split(
    dataset.sample(6000, random_state=42), train_size=0.80, random_state=42)

# train, validation and test datas
train_data, validation_data, test_data = split_data(train_df, test_df)
print(f"Total de imagens de treino: {train_data.samples}, Total de imagens de validação: {validation_data.samples}, Total de imagens de teste: {test_data.samples}")

# CNN architecture: 
model = tf.keras.models.Sequential()

# first convolution and MaxPooling Layers
model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = (3 , 3), input_shape = (227, 227, 3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

# second convolution and MaxPooling Layers
model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

# Dense and Flatten Layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

# showing model's summary
model.summary()

# compiling the model
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

# training the model
history = model.fit(
    train_data,
    validation_data = validation_data,
    epochs = 20
)

history = history.history

# Pickle the history to file
with open('cnn_model_history', 'wb') as f:
    pickle.dump(history, f)

# saving the model to cnn_model.h5 file
model.save('cnn_model.h5')
