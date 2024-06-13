# importing necessary libraries
import sys
sys.path.append('surface_crack_detection/')
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils.utils import split_data
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay 

#load dataset
dataset = pd.read_csv('surface_crack_detection\classification\dataset_binary\dataset_final.csv')

#split dataset
train_df, test_df = train_test_split(
    dataset.sample(frac = 1.0, random_state=42), train_size=0.80, random_state=42)

# train, validation and test datas
train_data, validation_data, test_data = split_data(train_df, test_df)
print(f"\nTotal de imagens de treino: {train_data.samples}, Total de imagens de validação: {validation_data.samples}, Total de imagens de teste: {test_data.samples}")

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
    epochs = 100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
            )
        ]
)

history = history.history

# Pickle the history to file
with open('surface_crack_detection/classification/models/historys/cnn_classification_model_history', 'wb') as f:
    pickle.dump(history, f)

# saving the model to cnn_model.h5 file
model.save('surface_crack_detection/classification/models/trained/cnn_classification_model.h5')

# curves

plt.figure(figsize = (10, 4))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Training", "Accuracy"])

plt.subplot(1, 2, 2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Training Loss", "Validation Loss"])

# saving the curves
plt.savefig("surface_crack_detection/classification/models/figures/CNN-curves.jpg")

# testing the model

model_prediction = model.predict(test_data)

# testing
predictions = np.squeeze(model_prediction >= 0.5).astype(np.int32)
predictions = predictions.reshape(-1, 1)

results = model.evaluate(test_data)

# assigning the results into loss and accuracy
loss = results[0]
accuracy = results[1]

# showing up the results
print(f"Model's accuracy: {(accuracy*100):0.2f}%")
print(f"Model's loss: {(loss):0.2f}%")

# creating the confusion matrix
matrix = confusion_matrix(test_data.labels, predictions)
classifications = classification_report(test_data.labels, predictions, target_names = ["ISOLATED", "DISSEMINATED"])
display = ConfusionMatrixDisplay(matrix)

# ploting the Matrix
display.plot()

plt.xticks(ticks = np.arange(2), labels = ["ISOLATED", "DISSEMINATED"])
plt.yticks(ticks = np.arange(2), labels = ["ISOLATED", "DISSEMINATED"])

plt.xlabel("Predict")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

# saving the confusion matrix
plt.savefig("surface_crack_detection/classification/models/figures/CNN-classification-confusion-matrix.jpg")