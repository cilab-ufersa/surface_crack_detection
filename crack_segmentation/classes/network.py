import sys


class Network:
    def __init__(
        self, args, IMAGE_DIMS, N_FILTERS, BS, INIT_LR, opt, loss, metrics
    ):
        self.args = args
        self.IMAGE_DIMS = IMAGE_DIMS
        self.N_FILTERS = N_FILTERS
        self.BS = BS
        self.INIT_LR = INIT_LR
        self.opt = opt
        self.loss = loss
        self.metrics = metrics

    def define_network(self):
        sys.path.append(self.args['main'])

        if self.args['model'] == 'unet':
            from model import unet

            model = unet(self.IMAGE_DIMS, num_filters=self.N_FILTERS)

        elif 'sm' in self.args['model']:
            #import segmentation-model as sm

            _, model_to_use, BACKBONE = self.args["model"].split(
                '_')  # ['sm', 'FPN', 'mobilenet']

            # definindo os parâmetros da rede
            num_classes = 1
            activation = 'sigmoid'
            # None or 'imagenet'
            encoder_weights = self.args['encoder_weights']

            if model_to_use == 'FPN':
                pyramid_block_filters = 512

                model = sm.FPN(
                    BACKBONE,
                    input_shape=self.IMAGE_DIMS,
                    classes=num_classes,
                    activation=activation,
                    encoder_weights=encoder_weights,
                    pyramid_block_filters=pyramid_block_filters,
                    pyramid_dropout=self.args['dropout']
                )

            elif model_to_use == 'Unet':
                model = sm.Unet(
                    BACKBONE,
                    input_shape=self.IMAGE_DIMS,
                    classes=num_classes,
                    activation=activation,
                    encoder_weights=encoder_weights,
                    decoder_filters=(1024, 512, 256, 128, 64),
                    # Dropout=self.args['dropout']
                )

        elif self.args['model'] == 'Deeplabv3':
            sys.path.append(self.args["main"] + 'networks/')

            from model import Deeplabv3

            weights = 'pascal_voc'
            input_shape = self.IMAGE_DIMS
            classes = 1
            BACKBONE = 'xception'  # 'xception','mobilenetv2'
            activation = 'sigmoid'  # One of 'softmax', 'sigmoid' or None
            OS = 16  # {8,16}

            model = Deeplabv3(weights=weights, input_shape=input_shape, classes=classes, backbone=BACKBONE,
                              OS=OS, activation=activation)

            import tensorflow as tf
            self.opt = tf.keras.optimizers.Adam(self.INIT_LR)

        elif self.args['model'] == 'DeepCrack':

            sys.path.append(self.args["main"] + 'networks/crack-detection')

            from edeepcrack_cls import Deepcrack

            model = Deepcrack(input_shape=(
                self.BS, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2]))

            import tensorflow as tf
            self.opt = tf.keras.optimizers.Adam(self.INIT_LR)

        model.compile(optimizer=self.opt, loss=self.loss,
                      metrics=[self.metrics])

        return model
