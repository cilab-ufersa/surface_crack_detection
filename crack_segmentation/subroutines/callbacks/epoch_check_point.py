import os
from keras.callbacks import Callback

class EpochCheckpoint(Callback):
    def __init__(
        self,
        outputPath_checkpoints,
        outputPath_weights,
        save_model_weights,
        every=5,
        startAt=0,
        info='',
        counter='',
        extension='.h5'
    ):
        super(Callback, self).__init__()

        self.outputPath_checkpoints = outputPath_checkpoints
        self.outputPath_weights = outputPath_weights
        self.save_model_weights = save_model_weights
        self.every = every
        self.intEpoch = startAt
        self.info = info
        self.counter = counter
        self.extension = extension

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if (self.intEpoch + 1) % self.every == 0:

            # check whether to save the whole model or only the weight
            if self.save_model_weights == 'model':
                folder_output = self.outputPath_checkpoints

            elif self.save_model_weights == 'weights':
                folder_output = self.outputPath_weights

            # define the name of saved model/weights
            if self.info == '':
                p = os.path.sep.join(
                    [folder_output, "{}_epoch_{}{}".format(
                        self.counter, self.intEpoch + 1, self.extension)]
                )
            else:
                p = os.path.sep.join(
                    [folder_output, "{}_{}_epoch_{}{}".format(
                        self.info, self.counter, self.intEpoch + 1, self.extension)]
                )

            # check whether to save the whole model or only the weight
            if self.save_model_weights == 'model':
                self.model.save(p, overwrite=True)
            elif self.save_model_weights == 'weights':
                self.model.save_weights(p)

        self.intEpoch += 1
