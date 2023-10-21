from utils import save_dataset
from pathlib import Path
import pandas as pd


dataset = save_dataset(paths=["dataset/Positive/", "dataset/Negative/"], filename="dataset/dataset_final.csv")

print(dataset.head())
