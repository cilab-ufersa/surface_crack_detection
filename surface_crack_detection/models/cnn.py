import sys 
sys.path.append('surface_crack_detection')
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils.utils import split_data

#load dataset
dataset = pd.read_csv('dataset/dataset_final.csv')

#split dataset
train_df, test_df = train_test_split(
    dataset.sample(6000, random_state=42), train_size=0.80, random_state=42)


train_data, validation_data, test_data = split_data(train_df, test_df)
print(f"Total de imagens de treino: {train_data.samples}, Total de imagens de validação: {validation_data.samples}, Total de imagens de teste: {test_data.samples}")



