# bibliotecas necessárias
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf


def generate_df(image_dir, label):
    """
    Gera um dataframe com os dados das imagens

    Args:
        image_dir (str): caminho da pasta com as imagens
        label (str): rótulo da imagem

    Returns:
        df (pd.DataFrame): dataframe com os dados das imagens
    """

    filepaths = pd.Series(list(image_dir.glob(r"*.jpg")), name="Filepath").astype(str)
    labels = pd.Series(label, name="Label", index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)

    return df


def save_dataset(
    paths=["dataset/Positive", "dataset/Negative"], filename="dataset_final.csv"
):
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

    positive_df = generate_df(positive_dir, label="POSITIVE")
    negative_df = generate_df(negative_dir, label="NEGATIVE")

    dataset = (
        pd.concat([positive_df, negative_df], axis=0)
        .sample(frac=1.0, random_state=1)
        .reset_index(drop=True)
    )
    dataset.to_csv(filename, index=False)

    return dataset


def split_data(
    train_df,
    test_df,
    image_width=227,
    image_height=227,
    image_channels=255.0,
    classes_names=["Sem fissura", "Com fissura"],
    class_mode="binary",
    validation_split=0.2,
    preprocess_input=None,
):
    """
    Divide o dataset em treino, validação e teste

    Args:
        train_df (pd.DataFrame): dataframe com os dados das imagens de treino
        test_df (pd.DataFrame): dataframe com os dados das imagens de teste
        image_width (int): largura da imagem
        image_height (int): altura da imagem
        image_channels (int): número de canais da imagem
        classes_names (list): lista com os nomes das classes
        class_mode (str): modo de classificação, binário ou categórico
        validation_split (float): proporção de imagens para validação
        preprocess_input (None | list): preprocessa um tensor ou um array Numpy que codifica um lote de imagens

    Returns:
        train_data (pd.DataFrame): dataframe com os dados das imagens de treino
        valid_data (pd.DataFrame): dataframe com os dados das imagens de validação
        test_data (pd.DataFrame): dataframe com os dados das imagens de teste
    """

    image_size = (image_width, image_height)

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / image_channels,
        validation_split=validation_split,
        preprocessing_function=preprocess_input,
    )
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / image_channels, preprocessing_function=preprocess_input
    )

    train_data = train_gen.flow_from_dataframe(
        train_df,
        x_col="Filepath",
        y_col="Label",
        target_size=image_size,
        color_mode="rgb",
        class_mode=class_mode,
        batch_size=32,
        shuffle=False,
        seed=42,
        subset="training",
    )

    valid_data = train_gen.flow_from_dataframe(
        train_df,
        x_col="Filepath",
        y_col="Label",
        target_size=image_size,
        color_mode="rgb",
        class_mode=class_mode,
        batch_size=32,
        shuffle=False,
        seed=42,
        subset="validation",
    )

    test_data = test_gen.flow_from_dataframe(
        test_df,
        x_col="Filepath",
        y_col="Label",
        target_size=image_size,
        color_mode="rgb",
        class_mode=class_mode,
        batch_size=32,
        shuffle=False,
        seed=42,
    )

    return train_data, valid_data, test_data


def plot_confusion_matrix(y_true, y_pred, labels=["Negative", "Positive"], title=""):
    """
    Plota a matriz de confusão

    Args:
        y_true (np.array): array com os valores reais
        y_pred (np.array): array com os valores previstos
        labels (list): lista com os nomes das classes
        title (str): título do gráfico

    Returns:
        disp (ConfusionMatrixDisplay): matriz de confusão
    """

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
    disp.ax_.set_xlabel("Prediction")
    disp.ax_.set_ylabel("Real")

    return disp
