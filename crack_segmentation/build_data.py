import sys
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from imutils import paths
from classes.Config_class import Config
from skimage.transform import resize

from subroutines.hdf5 import WriterMasks

folder = {}

folder['initial'] = 'crack_segmentation/'
folder['main'] = folder['initial'] + ''

# if folder['main'] == '', then the current working directory will be used
if folder['main'] == '':
    folder['main'] = os.getcwd()

sys.path.append(folder["main"])

cnf = Config(folder['main'])
args = cnf.set_repository()

IMAGE_DIMS = cnf.IMAGE_DIMS

data_path = list(paths.list_images(args['images']))
label_path = list(paths.list_images(args['masks']))

split = train_test_split(data_path, label_path, test_size=cnf.TEST_SIZE)

(train_x, valid_x, train_y, valid_y) = split

dataset = [
    ('train', train_x, train_y, args['TRAIN_HDF5']),
    ('validation', valid_x, valid_y, args['VAL_HDF5'])
]

for (dType, images_path, masks_path, output_path) in dataset:
    writer = WriterMasks(
        (len(images_path), IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2]),
        output_path
    )

    # percorrendo o caminho das imagens
    for (ii, (im_path, mask_path)) in enumerate(zip(images_path, masks_path)):
        image = cv2.imread(im_path)

        if IMAGE_DIMS != image.shape:
            image = resize(image, (IMAGE_DIMS),
                           mode='constant', preserve_range=True)

        image = image / 255

        mask = cv2.imread(mask_path, 0)

        if IMAGE_DIMS[0:2] != mask.shape:
            mask = resize(
                mask, (IMAGE_DIMS[0], IMAGE_DIMS[1]), mode='constant', preserve_range=True)

        mask = np.expand_dims(mask, axis=-1)
        mask = mask / 255

        writer.add([image], [mask])

    writer.close()
