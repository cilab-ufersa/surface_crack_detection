import os
import h5py

class WriterMasks:
    def __init__(
        self,
        dims,
        output_path,
        data_key='images',
        buffer_size=1000
    ):
        if os.path.exists(output_path):
            raise ValueError(
                "The supplied ‘outputPath‘ already "
                "exists and cannot be overwritten. Manually delete "
                "the file before continuing.", output_path
            )
        # abrindo um database hdf5 para escrever e criar dois dataset:
        # um para armazenar as imagens/features e outro
        # para armazenar as classe labels
        self.database = h5py.File(output_path, "w")

        self.dataset = self.database.create_dataset(
            data_key, dims, dtype='float'
        )

        self.labels = self.database.create_dataset(
            "labels", dims[:-1] + (1,), dtype='float'
        )

        self.buf_size = buffer_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.buf_size:
            self.flush()

    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.dataset[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.database.close()
