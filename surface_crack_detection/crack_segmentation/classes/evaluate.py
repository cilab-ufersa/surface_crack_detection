import sys
import tensorflow as tf
from keras.models import model_from_json


class LoadModel:
    def __init__(self, args, IMAGE_DIMS, BS):
        self.args = args
        self.IMAGE_DIMS = IMAGE_DIMS
        self.BS = BS

    def load_pretrained_model(self):
        """
        Load a pretrained model
        """

        # Load pretrained DeepCrack
        if self.args["model"] == 'DeepCrack':

            sys.path.append(self.args["main"] + 'networks/')
            from edeepcrack_cls import Deepcrack

            model = Deepcrack(input_shape=(
                self.BS, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2]))

            # load weights into new model
            model.load_weights(
                self.args['weights'] + self.args['pretrained_filename'])

        # Load pretrained model
        # This option is not supported for the current version of the code for the 'evaluation' mode
        # Print an explanatory comment and exit
        elif self.args['save_model_weights'] == 'model':

            raise ValueError("The option to load a model is not supported for the 'evaluation' mode." +
                             "In case you need to use the pretraine model to perform predictions, then" +
                             "train the model with the option: args['save_model_weights'] == 'weights'" +
                             "\nThe analysis will be terminated")

        # Load model from JSON file and then load pretrained weights
        else:

            # If pretrained Deeplabv3 will be loaded, import the Deeplabv3 module
            if self.args["model"] == 'Deeplabv3':
                sys.path.append(self.args["main"] + 'networks/')
                from network import Deeplabv3

            # load json and create model
            json_file = open(self.args['model_json'], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            try:
                model = tf.keras.models.model_from_json(loaded_model_json)
            except:
                model = tf.keras.models.model_from_json(loaded_model_json)

            # load weights into new model
            model.load_weights(
                self.args['weights'] + self.args['pretrained_filename'])

        return model
