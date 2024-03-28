import sys
import keras


class Optimizer:
    def __init__(self, args, INIT_LR):
        self.args = args
        self.INIT_LR = INIT_LR

    def define_Optimizer(self):

        sys.path.append(self.args["main"])

        if self.args['optimizer'] == 'Adam':
            from keras.optimizers import Adam
            opt = Adam(self.INIT_LR)

        elif self.args['optimizer'] == 'SGD':
            from keras.optimizers import SGD
            opt = SGD(self.INIT_LR)

        elif self.args['optimizer'] == 'RMSprop':
            from keras.optimizers import RMSprop
            opt = RMSprop(self.INIT_LR)

        return opt
