import numpy as np
import h5py

class GeneratorMask:
    def __init__(
        self,
        dbPath,
        batchSize,
        preprocessors=None,
        shuffle=False, aug=None,
        binarize=True,
        classes=2,
        threshold=0.5
    ):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.shuffle = shuffle

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath, 'r+')
        self.numImages = self.db["labels"].shape[0]
        # threshold to binarize mask
        self.threshold = threshold

    def generator(self, passes=np.inf):
        epochs = 0

        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

                if self.preprocessors is not None:
                    procImages = []

                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        procImages.append(image)

                    images = np.array(procImages)

                if self.aug is not None:
                    seed = 2018

                    image_generator = self.aug.flow(
                        images,
                        seed=seed,
                        batch_size=self.batchSize,
                        shuffle=self.shuffle
                    )

                    mask_generator = self.aug.flow(
                        labels,
                        seed=seed,
                        batch_size=self.batchSize,
                        shuffle=self.shuffle
                    )

                    train_generator = zip(image_generator, mask_generator)
                    (images, labels) = next(train_generator)

                    if self.binarize:
                        for ii in range(0, len(labels)):
                            labels[ii] = np.where(
                                labels[ii] > self.threshold, 1., 0.)

                yield (images, labels)

                epochs += 1

    def close(self):
        self.db.close()
