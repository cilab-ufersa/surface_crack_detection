import sys
sys.path.append("surface_crack_detection")

import numpy as np
import cv2
from keras.models import load_model


def classify(model_name, img_path):
    try:
        # Load model
        model = load_model(f'surface_crack_detection/models/trained/{model_name}_model.h5')

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Resize based on model requirements
        if model_name == "vgg":
            img = cv2.resize(img, (150, 150))
        else:
            img = cv2.resize(img, (227, 227))

        # Preprocess image
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # Make prediction
        y_pred = np.argmax(model.predict(img), axis=-1)

        # Print results
        print(f"modelo: {model_name}")
        print("positive" if y_pred[0] == 1 else "negative")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ./surface_crack_detection/predictions.py <model> <image_path>")
        sys.exit(1)

    classify(sys.argv[1], sys.argv[2])