import os
import sys


class Config:
    def __init__(self, working_folder):
        self.working_folder = working_folder
        self.mode = 'train'  # 'train', 'evaluate' or 'build_data'
        self.info = 'crack_detection'
        self.IMAGE_DIMS = (224, 224, 3)
        self.batch_size = 4
        self.epochs = 3
        self.learning_rate = 0.0005

        # The parameters of the configuration used will be stored in the dictionary args
        self.args = {}
        # 'Deeplabv3', 'Unet', 'DeepCrack', 'sm_Unet_mobilenet'
        self.args["model"] = "sm_Unet_mobilenet"
        self.args['regularization'] = 0.001
        self.args['optimizer'] = 'Adam'  # 'SGD' or 'Adam' or 'RMSprop'
        self.args['aug'] = False  # True or False
        # dropout: None or insert a value (e.g. 0.5)
        self.args['dropout'] = 0.5
        self.args['batchnorm'] = True  # True or False
        self.args['save_model_weights'] = 'weights'  # 'model' or 'weights'

        # Parameters to define for the configuration of Unet
        self.N_FILTERS = 64
        self.args['kernel_init'] = 'he_normal'
        self.args['encoder_weights'] = 'imagenet'  # None or 'imagenet'

        self.args['loss'] = 'WCE'
        if self.args['loss'] == 'Focal_Loss':
            # alpha value when focal loss is used default: 0.25
            self.args['focal_loss_a'] = 0.25
            # gamma value when focal loss is used. default: 2.
            self.args['focal_loss_g'] = 2
        elif self.args['loss'] == 'WCE':
            # weight of the positive class (i.e. crack class)
            self.args['WCE_beta'] = 10

        self.args['start_epoch'] = 0
        self.args['every'] = 5

        self.args['metric_to_plot'] = 'F1_score_dil'

        self.args['binarize'] = True  # True or False

        self.TEST_SIZE = 0.40

    def check_folder_exists(self, folder_path):
        """
        check if folder exists and if not create it
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def set_repository(self):
        self.args['main'] = self.working_folder
        self.args['dataset'] = self.args['main'] + 'dataset/'

        # diretório para salvar as saídas
        self.args['output'] = self.args['main'] + 'output/'
        # a saída será salva no formato hdf5
        self.args['hdf5'] = self.args['output'] + 'hdf5/'
        # diretório onde o modelo será salvo
        self.args['checkpoints'] = self.args['output'] + 'checkpoints/'
        # diretório onde os pesos serão salvo
        self.args['weights'] = self.args['output'] + 'weights/'
        self.args['model_json_folder'] = self.args['output'] + 'model_json/'
        self.args['predictions'] = self.args['output'] + 'predictions/'

        # criando os diretórios
        folders = [
            self.args['hdf5'],
            self.args['checkpoints'],
            self.args['weights'],
            self.args['model_json_folder'],
            self.args['predictions']
        ]

        for f in folders:
            self.check_folder_exists(f)

        # definindo as saídas dos arquivos hdf5
        self.args['TRAIN_HDF5'] = self.args['hdf5'] + 'train.hdf5'
        self.args['VAL_HDF5'] = self.args['hdf5'] + 'valid.hdf5'

        self.args['EVAL_HDF5'] = self.args['hdf5'] + 'valid.hdf5'

        # definindo os caminhos onde das imagens e máscaras
        self.args['images'] = self.args['dataset'] + \
            '{}_{}_images/'.format(self.info, self.IMAGE_DIMS[0])
        self.args['masks'] = self.args['dataset'] + \
            '{}_{}_masks/'.format(self.info, self.IMAGE_DIMS[0])

        if self.mode == 'train':
            # Path to the counter file
            self.args['counter_file'] = self.args['output'] + 'counter.txt'

            # If 'counter.txt' exists read the counter
            if os.path.exists(self.args['counter_file']):
                # Ask whether to change counter. acceptable answers are 'y' or 'n'
                # If anything else is given as input, ask again
                while True:
                    self.args['counter_check'] = input(
                        'Shall I change the counter [y/n]:')
                    if self.args['counter_check'] == 'y' or self.args['counter_check'] == 'n':
                        break
                    else:
                        continue

                # If input was 'y', change the counter
                if self.args['counter_check'] == 'y':
                    file = open(self.args['counter_file'], 'r')
                    self.args['counter'] = int(file.read())+1
                    file.close()
                    file = open(self.args['counter_file'], 'w')
                    file.write(str(self.args['counter']))
                    file.close()
                # If input was 'n', use the same counter
                else:
                    file = open(self.args['counter_file'], 'r')
                    self.args['counter'] = int(file.read())
                    file.close()

            # If the counter file doesn't exist, use the os.getpid() as counter
            else:
                self.args['counter'] = os.getpid()

            # Print the counter
            print(self.args['counter'])

            # Store results (i.e. metrics, loss) to CSV format
            # Check if the counter value was used before
            # If it was used before ask the user whether to proceed
            # If 'n' is passed, the analysis will be terminated
            self.args['CSV_PATH'] = self.args['output'] + \
                '{}_{}.out'.format(self.info, self.args['counter'])
            if os.path.exists(self.args['CSV_PATH']):
                print(
                    "The counter '{}' has been used before\nShould the analysis continue [y/n]:".format(self.args['counter']))
                check = input()

                if check == 'n':
                    print('The analysis will be terminated')
                    sys.exit(1)

            # armazenando os resultados em um arquivo csv
            self.args['CSV_PATH'] = self.args['output'] + \
                '{}_{}.out'.format(self.info, self.args['counter'])
            self.args['FIG_PATH'] = self.args['output'] + \
                self.info + '_{}.png'.format(self.args['counter'])
            self.args['JSON_PATH'] = self.args['output'] + \
                self.info + '_{}.json'.format(self.args['counter'])
            self.args['architecture'] = self.args['output'] + \
                '{}_architecture_{}.png'.format(
                    self.info, self.args['counter'])
            self.args['summary'] = self.args['output'] + \
                '{}_summary_{}.txt'.format(self.info, self.args['counter'])

        elif self.mode == 'evaluate':
            self.args['counter'] = 1
            self.args['pretrained_filename'] = 'crack_detection_1_epoch_2_F1_score_dil_0.111.h5'
            self.args['predictions_subfolder'] = '{}{}/'.format(
                self.args['predictions'], self.args['pretrained_filename'])
            self.args['predictions_dilate'] = True

        if (self.mode == 'train') or (self.mode == 'evaluate'):
            self.args['model_json'] = self.args['model_json_folder'] + \
                self.info + '_{}.json'.format(self.args['counter'])

        return self.args
