import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from PIL import Image


def white_pixels(binary_img):

    white_pixels = np.where(binary_img == 255)


    x_coords = white_pixels[1]
    y_coords = white_pixels[0]
    
    return x_coords, y_coords

def fit_line(ax, binary_img, color, value_min=180, value_max=200):
     
    x_coords, y_coords = white_pixels(binary_img)

    coeffs = np.polyfit(x_coords, y_coords, 1)
    poly_func = np.poly1d(coeffs)
    arctan = np.arctan(coeffs[0])
    degree = np.degrees(arctan)

    x_values = np.linspace(min(x_coords), max(x_coords), len(x_coords))

    ax.scatter(x_coords, y_coords, c='k', marker='o')
    ax.plot(x_values, poly_func(x_values), c=color, label=f"Angle: {(degree * (- 1)):.2f}")

    ax.legend(fontsize=16)

    ax.set_xlim(0, max(x_coords))
    ax.set_ylim(min(y_coords),max(y_coords))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
    

def count_white_segments(binary_img, filename):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_white_segments = len(contours)
    print(f"Number of cracks: {num_white_segments}")

    fig, ax = plt.subplots(figsize=(5,5))
    color = ['r', 'g', 'c', 'b', 'm', 'y', 'orange', 'brown', 'pink']
    
    value =[180, 190]
    value_max = [210, 220]
    
    
    for i, contour in enumerate(contours):
        blank_image = np.zeros_like(binary_img)
        segment = cv2.drawContours(blank_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        fit_line(ax, segment, color[i], value[1], value_max[1])

    ax.set_xlim(0, binary_img.shape[1])
    ax.set_ylim(240, 0)
    ax.set_ylabel('Pixels', fontsize=16)
    ax.set_xlabel('Pixels', fontsize=16)
   
    plt.savefig(filename, format='eps', bbox_inches='tight', pad_inches=0)
    plt.show()

def show_results(img_path):

    image = Image.open(img_path)
    binary_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    axs = plt.subplots(1, 2, figsize=(5, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Binary Image')
    
    filename = img_path.split('.')[0] + '.eps'
    print(filename)
    
    count_white_segments(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), filename=filename)

def save_points(binary_img, filename):
    x_coords, y_coords = white_pixels(binary_img)

    with open(filename, 'w') as f:
        f.write('x,y\n')
        for x, y in zip(x_coords, y_coords):
            f.write(f'{x},{y}\n')

    print(f"Points saved to {filename}")

