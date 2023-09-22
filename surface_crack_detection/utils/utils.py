# bibliotecas necessárias
import pandas as pd

# função que gerará os dataframes 
def generate_df(image_dir, label):

    # diretório das imagens
    filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name = 'Filepath').astype(str)
    
    # nome atribuído a cada imagem do diretório
    labels = pd.Series(label, name = 'Label', index = filepaths.index)
    
    # criando o dataframe através da concatenação do caminho e dos labels das imagens
    df = pd.concat([filepaths, labels], axis = 1)
    
    return df