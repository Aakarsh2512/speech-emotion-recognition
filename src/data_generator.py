"""
Feeds training batches to Keras and applies SpecAugment on the fly,
so the model sees a slightly different version of every spectrogram each epoch.
"""
import math

import keras
import numpy as np

from src.augment import spec_augment


class SpecAugmentGenerator(keras.utils.PyDataset):

    def __init__(self, mels, labels, batch_size=32, spec_augment_probability=0.8, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.mels = mels                    # already normalised, shape (N, 64, 94)
        self.labels = labels
        self.batch_size = batch_size
        self.spec_augment_probability = spec_augment_probability
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(mels))
        self.rng.shuffle(self.order)

    def __len__(self):
        return math.ceil(len(self.mels) / self.batch_size)

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        end = min(start + self.batch_size, len(self.mels))
        batch_ids = self.order[start:end]

        batch_x = np.zeros((len(batch_ids), self.mels.shape[1], self.mels.shape[2], 1), dtype=np.float32)
        batch_y = self.labels[batch_ids]

        for position in range(len(batch_ids)):
            spectrogram = self.mels[batch_ids[position]]
            if self.rng.random() < self.spec_augment_probability:
                spectrogram = spec_augment(spectrogram, self.rng, fill_value=0.0)
            batch_x[position, :, :, 0] = spectrogram

        return batch_x, batch_y

    def on_epoch_end(self):
        # new random order every epoch
        self.rng.shuffle(self.order)
