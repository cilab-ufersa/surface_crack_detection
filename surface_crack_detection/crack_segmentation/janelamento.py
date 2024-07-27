from PIL import Image
import os
import numpy as np
import cv2

def janelamento(img_path):
    image = Image.open(img_path)
    image = np.array(image)

    height, width, _ = image.shape
    window_size = (224, 224)
    
    num_rows = height // window_size[0]
    num_cols = width // window_size[1]
    final_image = np.zeros((num_rows * window_size[0], num_cols * window_size[1], 3), dtype=np.uint8)
    
    for contador in range(height // window_size[0]):
        for contador2 in range(width // window_size[1]):
            x = contador * window_size[0]
            y = contador2 * window_size[1]
            window = image[x:x + window_size[0], y:y + window_size[1]]
            cv2.imwrite(f'surface_crack_detection/crack_segmentation/img/windows_2/window_{contador2}_{contador}.jpg', window)
                

    print("Janelamento concluído com sucesso!")

    
def concatenacao(img_path):
    
    image = Image.open(img_path)
    image = np.array(image)

    height, width, _ = image.shape
    window_size = (224, 224)
    
    # Determine the layout of the images
    num_rows = height // window_size[0]
    num_cols = width // window_size[1]
    
    # Create an empty array to hold the final image
    final_image = np.zeros((num_rows * window_size[0], num_cols * window_size[1]), dtype=np.uint8)
    
    # Ensure that we only open as many segmentations as there are windows in the original image
    #assert len(segmentations) >= num_rows * num_cols, "Not enough segmentations to fill the image."
 
    for row in range(num_rows):
        for col in range(num_cols):
            x = row * window_size[0]
            y = col * window_size[1]
            
            image = Image.open(f'surface_crack_detection/crack_segmentation/img/segmentations_2/window_{col}_{row}.jpg')
            window = np.array(image)
            
            final_image[x:x + window_size[0], y:y + window_size[1]] = window
   

    cv2.imwrite('surface_crack_detection/crack_segmentation/img/output/img_final_seg.jpg', final_image)
    
concatenacao('surface_crack_detection/crack_segmentation/img/04.jpg')
#janelamento('surface_crack_detection/crack_segmentation/img/04.jpg')
