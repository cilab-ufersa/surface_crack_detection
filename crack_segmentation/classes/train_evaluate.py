import sys
from classes.config import Config
from subroutines.hdf5 import GeneratorMask

folder = {}
folder['initial'] = '/content/drive/MyDrive/'
folder['main'] = folder['initial'] + 'segmentation/'

sys.path.append(folder["main"])

cnf = Config(folder['main'])
args = cnf.set_repository()

IMAGE_DIMS = cnf.IMAGE_DIMS
BS = cnf.batch_size
epochs = cnf.epochs
INIT_LR = cnf.learning_rate
N_FILTERS = cnf.N_FILTERS
info = cnf.info
mode = cnf.mode

if mode == 'train':
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import CSVLogger

    from subroutines.callbacks import EpochCheckpoint
    from subroutines.callbacks import TrainingMonitor
    from subroutines.visualize_model import visualize_model
    from metrics import Metrics
    from loss import Loss
    from optimizer import Optimizer
    from network import Network

    metrics = Metrics(args).define_Metrics()
    loss = Loss(args).define_Loss()
    opt = Optimizer(args, INIT_LR).define_Optimizer()
    model = Network(
        args, IMAGE_DIMS, N_FILTERS, BS, INIT_LR, opt, loss, metrics
    ).define_network()

    try:
        visualize_model(model, args['architecture'], args['summary'])
    except:
        from subroutines.visualize_model import visualize_model_tf
        visualize_model_tf(model, args['architecture'], args['summary'])

    if args['aug'] == True:
        aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                                 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                 horizontal_flip=True, fill_mode='nearest')
    else:
        aug = None

    # Load data generators
    trainGen = GeneratorMask(
        args['TRAIN_HDF5'], BS, aug=aug, shuffle=False, binarize=args['binarize'])
    valGen = GeneratorMask(
        args['VAL_HDF5'], BS, aug=aug, shuffle=False, binarize=args['binarize'])

    csv_logger = CSVLogger(args['CSV_PATH'], append=True, separator=';')

    # serialize model to JSON
    try:
        model_json = model.to_json()
        with open(args['model_json'], 'w') as json_file:
            json_file.write(model_json)
    except:
        pass

    temp = '{}_{}'.format(info, args['counter']) + "_epoch_{epoch}_" + \
        args['metric_to_plot'] + "_{val_" + args['metric_to_plot'] + ":.3f}.h5"

    if args['save_model_weights'] == 'model':
        ModelCheckpoint_file = args["checkpoints"] + temp
        save_weights_only = False

    elif args['save_model_weights'] == 'weights':
        ModelCheckpoint_file = args['weights'] + temp
        save_weights_only = True

    epoch_checkpoint = EpochCheckpoint(args['checkpoints'], args['weights'], args['save_model_weights'],
                                       every=args['every'], startAt=args['start_epoch'], info=info, counter=args['counter'])

    training_monitor = TrainingMonitor(args['FIG_PATH'], jsonPath=args['JSON_PATH'],
                                       startAt=args['start_epoch'], metric=args['metric_to_plot'])

    model_checkpoint = ModelCheckpoint(ModelCheckpoint_file, monitor='val_{}'.format(args['metric_to_plot']),
                                       verbose=1, save_best_only=True, mode='max', save_weights_only=save_weights_only)

    callbacks = [csv_logger, epoch_checkpoint,
                 training_monitor, model_checkpoint]

    history = model.fit(
        trainGen.generator(),
        steps_per_epoch=trainGen.numImages // BS,
        validation_data=valGen.generator(),
        validation_steps=valGen.numImages // BS,
        epochs=epochs,
        max_queue_size=BS * 2,
        callbacks=callbacks, verbose=1
    )

elif mode == 'evaluate':

    # load pretrained model/weights
    from evaluate import LoadModel
    model = LoadModel(args, IMAGE_DIMS, BS).load_pretrained_model()

    # Do not use data augmentation when evaluating model: aug=None
    evalGen = GeneratorMask(
        args['EVAL_HDF5'], BS, aug=None, shuffle=False, binarize=args['binarize'])

    # Use the pretrained model to fenerate predictions for the input samples from a data generator
    predictions = model.predict_generator(evalGen.generator(),
                                          steps=evalGen.numImages // BS+1, max_queue_size=BS * 2, verbose=1)

    # Define folder where predictions will be stored
    predictions_folder = '{}{}/'.format(args['predictions'],
                                        args['pretrained_filename'])
    # Create folder where predictions will be stored
    cnf.check_folder_exists(predictions_folder)

    # Visualize  predictions
    from subroutines.visualize_predictions import Visualize_Predictions
    Visualize_Predictions(args, predictions)
