import sys
sys.path.append('surface_crack_detection')
from utils.utils import split_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

# load dataset
dataset = pd.read_csv('dataset/dataset_final.csv')

# split dataset
train_df, test_df = train_test_split(dataset.sample(
    frac=1.0, random_state=42), train_size=0.80, random_state=42)

preprocess_input = tf.keras.applications.resnet.preprocess_input

# train, valid and test data
train_data, valid_data, test_data = split_data(
    train_df, test_df, image_channels=1.0, preprocess_input=preprocess_input)

# train, validation and test generator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.3, preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(160, 150),
    batch_size=64,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.3, preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_directory(
    'dataset',
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary',
    shuffle=True,
    subset='validation',
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.3, preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(150, 150),
    batch_size=32,
    color_mode='rgb',
    class_mode='binary',
    shuffle=False,
    seed=42
)

# train, validation and test data
train_data, valid_data, test_data = split_data(
    train_df, test_df, image_channels=1.0, preprocess_input=preprocess_input)

# using the ResNet50 architecture
pre_trained_model = tf.keras.applications.resnet50.ResNet50(
    include_top=False, weights='imagenet', input_shape=(150, 150, 3), pooling='avg')

# using this class at the end of an epoch
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.999):
            self.model.stop_training = True


callbacks = myCallback()

for layer in pre_trained_model.layers:
    layer.trainable = False

# building the model
x = pre_trained_model.output
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(5, activation='softmax')(x)

model = tf.keras.Model(pre_trained_model.input, output)

# compiling the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy'])

# showing model's summary
model.summary()

# training the model
history = model.fit(train_generator, validation_data=validation_generator,
                    epochs=10, verbose=1, callbacks=[callbacks])

# pickle the history to file
with open('surface_crack_detection/models/historys/resnet_model_history.h5', 'wb') as f:
    pickle.dump(history, f)

# saving the model to inception_model.h5 file
model.save('surface_crack_detection/models/trained/resnet_model.h5')

# curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy', 'Validation accuracy'])

plt.subplot(1, 2, 2)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Accuracy over time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training loss', 'Validation loss'])

# saving the curves
plt.savefig('surface_crack_detection/models/figures/resnet_model.jpg')

# making predictions
model_predictions = model.predict(test_generator)
predictions = np.squeeze(model_predictions >= 0.5).astype(np.int32)
predictions = predictions.reshape(-1, 1)

results = model.evaluate(test_generator)

# assigning the results into loss and accuracy
loss = results[0]
accuracy = results[1]

# showing up the results
print(f"Model's accuracy: {(accuracy*100):0.2f}%")
print(f"Model's loss: {(loss):0.2f}%")

# creating the confusion matrix
matrix = confusion_matrix(test_generator.labels, predictions)
classifications = classification_report(
    test_generator.labels, predictions, target_names=['WITHOUT_CRACK', 'WITH_CRACK'])
display = ConfusionMatrixDisplay(matrix)

display.plot()

plt.xticks(ticks=np.arange(2), labels=['WITHOUT_CRACK', 'WITH_CRACK'])
plt.yticks(ticks=np.arange(2), labels=['WITHOUT_CRACK', 'WITH_CRACK'])

plt.xlabel('Predict')
plt.ylabel('Actual')
plt.title('Confustion Matrix')

# saving the confusion matrix
plt.savefig(
    'surface_crack_detection/models/figures/resnet_matrix_confusion.jpg')
