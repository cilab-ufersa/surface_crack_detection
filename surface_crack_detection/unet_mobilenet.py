import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from crack_segmentation.subroutines.loss_metrics import (Weighted_Cross_Entropy, F1_score, F1_score_dil, Precision_dil)
from utils.utils import split_data

dataset = pd.read_csv('../dataset/dataset_final.csv')

dataset['Filepath'] = dataset['Filepath'].apply(lambda x: '../' + x)

train_df, test_df = train_test_split(dataset.sample(6000, random_state=42), train_size=0.8, random_state=42)
train_data, valid_data, test_data = split_data(train_df, test_df, image_width=224, image_height=224)

unet_model = tf.keras.models.load_model(
    'crack_segmentation/output/checkpoints/crack_detection_3_epoch_20_F1_score_dil_0.776.h5',
    custom_objects={
        'loss': Weighted_Cross_Entropy(10),
        'F1_score': F1_score,
        'F1_score_dil': F1_score_dil,
        'Precision_dil': Precision_dil
    })

unet_model.load_weights('crack_segmentation/output/weights/crack_detection_1_epoch_9_F1_score_dil_0.812.h5')

cnn_model = tf.keras.models.load_model('crack_segmentation/output/checkpoints/model99.h5')

for layer in unet_model.layers:
    layer.trainable = False

resized_unet_output = tf.keras.layers.Resizing(96, 96)(unet_model.output)
cnn_output = cnn_model(resized_unet_output)

combined_model = tf.keras.Model(inputs=unet_model.input, outputs=cnn_output)

combined_model.summary()

optimizer = tf.keras.optimizers.Adam()

combined_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

combined_model.fit(train_data, validation_data=valid_data, epochs=10)

combined_model.save('crack_segmentation/output/checkpoints/unet_mobilenet.h5')
