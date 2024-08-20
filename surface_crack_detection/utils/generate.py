from utils import *

dataset = save_dataset(paths=["surface_crack_detection\classification\masks_dataset\Disseminated", 
                              "surface_crack_detection\classification\masks_dataset\Isolated"],
                               first_label="Disseminated",
                               second_label="Isolated",
                               filename="surface_crack_detection\classification\masks_dataset\dataset_final.csv")

print(dataset.head())