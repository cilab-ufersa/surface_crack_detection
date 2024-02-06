import os
import tempfile
import keras
from keras.regularizers import l2
from keras.models import model_from_json


def add_regularization(model, regularization):
    regularizer = l2(regularization)

    # verificando se regularizer é uma instância da classe Regularizer
    if not isinstance(regularizer, keras.regularizers.Regularizer):
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    model_json = model.to_json()

    # salvando o modelo
    tmp_weigths_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weigths_path)

    # carregando o modelo
    model = model_from_json(model_json)

    # recarregando o modelo
    model.load_weights(tmp_weigths_path, by_name=True)

    return model
