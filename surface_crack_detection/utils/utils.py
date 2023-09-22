# bibliotecas necessárias
import pandas as pd
from pathlib import Path

# função que gerará os dataframes 


def generate_df(image_dir, label):

    """
    Gera um dataframe com os dados das imagens

    Args:
        image_dir (str): caminho da pasta com as imagens
        label (str): rótulo da imagem

    Returns:
        df (pd.DataFrame): dataframe com os dados das imagens
    """
    
    filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index) 
    df = pd.concat([filepaths, labels], axis=1)
    
    return df



def save_dataset(paths=["dataset/Positive", "dataset/Negative"], filename="dataset_final.csv"):
    """
    Salva o dataset em um arquivo csv

    Args:
        paths (list): lista com os caminhos das pastas com as imagens
        filename (str): nome do arquivo csv que será salvo

    Returns:
        dataset (pd.DataFrame): dataframe com os dados das imagens
    """

    positive_dir = Path(paths[0])
    negative_dir = Path(paths[1])

    positive_df = generate_df(positive_dir, label = 'POSITIVE')
    negative_df = generate_df(negative_dir, label = "NEGATIVE") 

    dataset = pd.concat([positive_df, negative_df], axis = 0).sample(frac = 1.0, random_state = 1).reset_index(drop=True)
    dataset.to_csv(filename, index = False)

    return dataset