import sys
sys.path.append("surface_crack_detection")

import numpy as np
import cv2
import os
from keras.models import load_model, Model
from crack_segmentation.subroutines.loss_metrics import *

loss = Weighted_Cross_Entropy(10)
precision_dil = Precision_dil
f1_score = F1_score
f1_score_dil = F1_score_dil

model = load_model(
    'surface_crack_detection/models/trained/unet_mobilenet.h5',
    custom_objects={
        'loss': loss,
        'Precision_dil': precision_dil,
        'F1_score': f1_score,
        'F1_score_dil': f1_score_dil
    }
)

def segmentation(path):
    """
    Gera predições de saída para as amostras de entrada
    
    Args:
        path (str): recebe o caminho das imagens para segmentação

    Returns:
        y_pred (numpy array): array contendo as predições
    """
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    segmented_output = Model(
        inputs=model.input, outputs=model.get_layer(name="sigmoid").output
    )

    y_pred = segmented_output.predict(np.expand_dims(img, axis=0))

    return y_pred

def classification(path):
    """Classifies the input image as containing a crack or not

    Args:
        Args:
        path (str): receives the path of the images for classification

    Returns:
        negative (float): probability of the image not containing a crack
        positive (float): probability of the image containing a crack
    """

    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    y_pred = model.predict(img)

    negative = y_pred[0][0] * 100
    positive = y_pred[0][1] * 100

    print('positive' if positive > negative else 'negative')

input_directory = "surface_crack_detection/image"
output_directory = "surface_crack_detection/image_output"

total_images_to_segment = len(input_directory)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

image_files = sorted(os.listdir(input_directory))

for i, image_name in enumerate(image_files):
    if i >= total_images_to_segment:
        break

    image_path = os.path.join(input_directory, image_name) 

    if os.path.exists(image_path):
        print(f"Image {i + 1}/{total_images_to_segment}")
        pred = segmentation(image_path)
        mask_name = f'{image_name.split(".")[0]}.jpg'
        mask_path = os.path.join(output_directory, mask_name)
        cv2.imwrite(mask_path, (pred[0] * 255.0).astype(np.uint8))

        classification(mask_path)