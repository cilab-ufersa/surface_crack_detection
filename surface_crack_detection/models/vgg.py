# importing necessary libraries
import sys
import pickle
sys.path.append('surface_crack_detection')
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.utils import split_data
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from keras.preprocessing.image import ImageDataGenerator


#defining our own Callback
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.999):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True

# defining the optimizer
RMS = tf.keras.optimizers.RMSprop

# defining VGG16
VGG = tf.keras.applications.VGG16

#load dataset
dataset = pd.read_csv('dataset/dataset_final.csv')

#split dataset
train_df, test_df = train_test_split(
    dataset.sample(frac = 1.0, random_state=42), train_size=0.80, random_state=42)

# train, validation and test datas
train_data, validation_data, test_data = split_data(train_df, test_df)
 

# train and validation generators
train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.3)
train_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size = (150, 150),
    batch_size = 64,
    class_mode = 'binary',
    shuffle = True,
    subset = 'training'
)

validation_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.3)
validation_generator = validation_datagen.flow_from_directory(
    'dataset/',
    target_size = (150, 150),
    batch_size = 64,
    class_mode = 'binary',
    shuffle = True,
    subset = 'validation'
)

# creating VGG16 model's instances from pre-trained weights
local_wights_file = "surface_crack_detection/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
pre_trained_model = VGG(input_shape = (150, 150, 3), include_top = False, weights = None)
pre_trained_model.load_weights(local_wights_file)

for layers in pre_trained_model.layers:
    layers.trainable = False

pre_trained_model.summary()

# defining the last trainable layer
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output

# adding our own layers
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

model_vgg = tf.keras.Model(pre_trained_model.input, x)

# compiling the model
model_vgg.compile(
    optimizer = RMS(learning_rate = 0.0001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

callbacks = MyCallback()

# training the model
history = model_vgg.fit(
    train_generator,
    validation_data = validation_generator,
    epochs = 7,
    verbose = 1,
    callbacks = [callbacks]
)

history = history.history

# Pickle the history to file
with open('surface_crack_detection/models/historys/vgg_model_history', 'wb') as f:
    pickle.dump(history, f)

# saving the model to vgg_model.h5 file
model_vgg.save("surface_crack_detection/models/trained/vgg_model.h5")

# training datas
accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(len(accuracy))

# ploting the curves

# accuracy curves
plt.figure(figsize = (10, 4))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Training", "Accuracy"])


# loss curves
plt.subplot(1, 2, 2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Training Loss", "Validation Loss"])

# saving the curves
plt.savefig("surface_crack_detection/models/figures/vgg-curves.jpg")

# testing the model
model_prediction = model_vgg.predict(test_data)

# testing
predictions = np.squeeze(model_prediction >= 0.5).astype(np.int32)
predictions = predictions.reshape(-1, 1)

results = model_vgg.evaluate(test_data)

# assigning the results into loss and accuracy
loss = results[0]
accuracy = results[1]

# showing up the results
print(f"Model's accuracy: {(accuracy*100):0.2f}%")
print(f"Model's loss: {(loss):0.2f}%")

# creating the confusion matrix
matrix = confusion_matrix(test_data.labels, predictions)
classifications = classification_report(test_data.labels, predictions, target_names = ["WITHOUT_CRACK", "WITH_CRACK"])
display = ConfusionMatrixDisplay(matrix)

# ploting the Matrix
display.plot()

plt.xticks(ticks = np.arange(2), labels = ["WITHOUT_CRACK", "WITH_CRACK"])
plt.yticks(ticks = np.arange(2), labels = ["WITHOUT_CRACK", "WITH_CRACK"])

plt.xlabel("Predict")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

# saving the confusion matrix
plt.savefig("surface_crack_detection/models/figures/VGG-confusion-matrix.jpg")