import sys

sys.path.append("surface_crack_detection")

import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from subroutines.loss_metrics import (
    Weighted_Cross_Entropy,
    F1_score,
    Precision_dil,
    F1_score_dil,
)
from utils.utils import split_data
from keras.layers import Resizing, Concatenate

dataset = pd.read_csv("dataset/dataset_final.csv")

train_df, test_df = train_test_split(
    dataset.sample(6000, random_state=42), train_size=0.80, random_state=42
)

preprocess_input = tf.keras.applications.resnet.preprocess_input

train_data, valid_data, test_data = split_data(
    train_df, test_df, image_width=224, image_height=224, 
    image_channels=1.0, preprocess_input=preprocess_input
)

# modelo de segmentação unet
unet_model = tf.keras.models.load_model(
    "surface_crack_detection/crack_segmentation/output/checkpoints/crack_detection_4_epoch_87_F1_score_0.755.h5",
    custom_objects={
        "loss": Weighted_Cross_Entropy(10),
        "F1_score": F1_score,
        "F1_score_dil": F1_score_dil,
        "Precision_dil": Precision_dil,
    },
)

# pesos da unet
unet_model.load_weights(
    "surface_crack_detection/crack_segmentation/output/weights/crack_detection_1_epoch_9_F1_score_dil_0.812.h5"
)

# modelo de classificação resnet
resnet_model = tf.keras.models.load_model(
    "surface_crack_detection/models/trained/resnet50_model.h5"
)

# desativando as camadas de treino da unet
for layer in unet_model.layers:
    layer.trainable = False

resized_unet_output = Concatenate()([unet_model.output, unet_model.output, unet_model.output])

resnet_output = resnet_model(resized_unet_output)

combined_model = tf.keras.models.Model(inputs=unet_model.input, outputs=resnet_output)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

combined_model.compile(optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

combined_model.summary()

combined_model.fit(train_data, validation_data=valid_data, epochs=100)

combined_model.save(
    "surface_crack_detection/crack_segmentation/output/checkpoints/unet_resnet50.h5"
)
