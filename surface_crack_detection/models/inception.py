import sys
sys.path.append('surface_crack_detection')
from utils.utils import split_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

# load dataset
dataset = pd.read_csv('dataset/dataset_final.csv')

# split dataset
train_df, test_df = train_test_split(
    dataset.sample(frac=1.0), test_size=0.80, random_state=42)

# train, validation and test data
train_data, valid_data, test_data = split_data(train_df, test_df)

# load the path to weights file
load_weights_file = 'surface_crack_detection/models/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# using the inceptionV3 architecture
base_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False, input_shape=(227, 227, 3), weights=load_weights_file)

# using this class at the end of an epoch
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.999):
            self.model.stop_training = True


callbacks = myCallback()

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

# compiling the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy',
              metrics=['accuracy'])

# showing model's summary
model.summary()


# training the model
history = model.fit(train_data, validation_data=valid_data,
                    epochs=7, verbose=1, callbacks=[callbacks])

# pickle the history to file
with open('surface_crack_detection/models/historys/inception_model_history.h5', 'wb',) as f:
    pickle.dumb(history, f)

# saving the model to inception_model.h5 file
model.save('surface_crack_detection/models/trained/inception_model.h5')

# curves
acc = history.history['accuracy']
val_acc = history.history['val_history']
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
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training loss', 'Validation loss'])

# saving the curves
plt.savefig('surface_crack_detection/models/figures/inception_curves.jpg')

# making predictions
model_prediction = model.predict(test_data)
predictions = np.squeeze(model_prediction >= 0.5).astype(np.int32)
predictions = predictions.reshape(-1,1)

results = model.evaluate(test_data)

# assigning the results into loss and accuracy
loss = results[0]
accuracy = results[1]

# showing up the results
print(f"Model's accuracy: {(accuracy*100):0.2f}%")
print(f"Model's loss: {(loss):0.2f}%")

# creating the confusion matrix
matrix = confusion_matrix(test_data.labels, predictions)
classifications = classification_report(test_data.labels, predictions, target_names=['WITHOUT_CRACK', 'WITH_CRACK'])
display = ConfusionMatrixDisplay(matrix)

display.plot()

plt.xticks(ticks=np.arange(2), labels=['WITHOUT_CRACK', 'WITH_CRACK'])
plt.yticks(ticks=np.arange(2), labels=['WITHOUT_CRACK', 'WITH_CRACK'])

plt.xlabel('Predict')
plt.ylabel('Actual')
plt.title('Confustion Matrix')

# saving the confusion matrix
plt.savefig('surface_crack_detection/models/figures/inception_matrix_confusion.jpg')
