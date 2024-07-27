import numpy as np
import cv2
import os
from keras.models import load_model
from subroutines.loss_metrics import *
from PIL import Image

loss = Weighted_Cross_Entropy(10)
precision_dil = Precision_dil
f1_score = F1_score
f1_score_dil = F1_score_dil

model = load_model(
    'surface_crack_detection/crack_segmentation/output/checkpoints/crack_detection_6_epoch_48_F1_score_0.784.h5',
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

    y_pred = model.predict(np.expand_dims(img, axis=0))

    return y_pred


input_directory = "surface_crack_detection/crack_segmentation/img/windows_2"
output_directory = "surface_crack_detection/crack_segmentation/img/segmentations_2"

total_images_to_segment = 234

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
