# importing necessary libraries
import sys
sys.path.append('surface_crack_detection')
from utils.utils import split_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

# load dataset
dataset = pd.read_csv('surface_crack_detection/classification/dataset_binary/dataset_final.csv')

# split dataset
train_df, test_df = train_test_split(dataset.sample(
    frac=1.0, random_state=42), train_size=0.80, random_state=42)

# input preprocessing
preprocess_input = tf.keras.applications.resnet.preprocess_input

# train, validation and test datas
train_data, valid_data, test_data = split_data(
    train_df, test_df, image_channels=1.0, preprocess_input=preprocess_input)

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.999):
            print('\nReached 99.9% accuracy sp cancelling training')
            self.model.stop_training = True


callbacks = MyCallback()

# resnet50 architecture
pre_trained_model = tf.keras.applications.resnet.ResNet50(
    include_top=False, weights='imagenet', input_shape=(227, 227, 3), pooling='avg')

for layer in pre_trained_model.layers:
    layer.trainable = False

# building the model
x = pre_trained_model.output
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(5, activation='softmax')(x)

model = tf.keras.Model(inputs=pre_trained_model.input, outputs=output)

# showing up model's summary
model.summary()

steps_per_epoch_training = len(train_data)
steps_per_epoch_validation = len(valid_data)

# compiling the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training the model
history = model.fit(train_data, validation_data=valid_data, epochs=30, verbose=1,
                    steps_per_epoch=steps_per_epoch_training, validation_steps=steps_per_epoch_validation, callbacks=[callbacks])

history = history.history

# pickle the history to file
with open('surface_crack_detection/classification/models/historys/resnet50_model_history', 'wb') as f:
    pickle.dump(history, f)

# saving the model
model.save('surface_crack_detection/classification/models/trained/resnet50_model.h5')

# curves
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Training and validation accuracy')
plt.legend(['Training accuracy', 'Validation accuracy'])

plt.subplot(1, 2, 2)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Training and validation loss')
plt.legend(['Training loss', 'Validation loss'])

# saving the curves
plt.savefig('surface_crack_detection/classification/models/figures/resnet50_curves.jpg')

# making predictions
model_predictions = model.predict(test_data)
predictions = np.argmax(model_predictions, axis=-1)

# testing the model
results = model.evaluate(test_data)

loss = results[0]
accuracy = results[1]

print(f"\nmodel's accuracy: {(accuracy*100):0.2f}%")
print(f"\nmodel's loss: {(loss*100):0.2f}%")

# creating confunsion matrix
matrix = confusion_matrix(test_data.labels, predictions)
classification = classification_report(
    test_data.labels, predictions, target_names=["ISOLATED", "DISSEMINATED"])
display = ConfusionMatrixDisplay(matrix)

display.plot()

plt.xticks(ticks=np.arange(2), labels=["ISOLATED", "DISSEMINATED"])
plt.yticks(ticks=np.arange(2), labels=["ISOLATED", "DISSEMINATED"])

plt.ylabel('Actual')
plt.xlabel('Predict')

plt.title('Confustion matrix')

# saving the confusion matrix
plt.savefig(
    'surface_crack_detection/classification/models/figures/resnet50_confusion_matrix.jpg')
