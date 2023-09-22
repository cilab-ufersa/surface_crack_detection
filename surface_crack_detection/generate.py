from utils import generate_df
from pathlib import Path
import pandas as pd

# caminho relativo das imagens
positive_dir = Path("../dataset/Positive/")
negative_dir = Path("../dataset/Negative/")

# dataframe positivos e negativos
positive_df = generate_df(positive_dir, label = 'POSITIVE')
negative_df = generate_df(negative_dir, label = "NEGATIVE") 

# criando um único datafrem
all_df = pd.concat([positive_df, negative_df], axis = 0).sample(frac = 1.0, random_state = 1).reset_index(drop=True)
print(all_df)