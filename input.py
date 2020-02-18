import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


class DataPreprocessing(object):
    """docstring for DataPreprocessing."""

    def __init__(self, input, **kwargs):
        super(DataPreprocessing, self).__init__()
        self.x = input
        self.n_samples = len(self.x)
        self.y = np.array(kwargs.get('labels', []), dtype='int32')
        self.m = np.array(kwargs.get('metadata', []), dtype='float32')
        self.l = np.array([len(x) for x in self.x], dtype='int32')
        self.maxlen = tf.reduce_max([len(x) for x in self.x])

    def pad_series(self, value=0):
        self.x = pad_sequences(sequences=self.x,
                               maxlen=self.maxlen,
                               dtype='float32',
                               padding='post',
                               value=value)

    def standardize(self):
        mean, std = tf.nn.moments(tf.cast(self.x, 'float32'), axes=[1])
        self.x = (self.x - tf.expand_dims(mean,1))/tf.expand_dims(std,1)

    def train_test_val_split(self, val=0.25, test=0.25):
        indices = np.arange(0, self.n_samples)
        shuffled_indices = np.random.shuffle(indices)

        self.x = self.x[shuffled_indices][0]
        self.y = self.y[shuffled_indices][0]
        self.l = self.l[shuffled_indices][0]

        train_ind = int(self.n_samples*(1-(val+test)))
        val_ind   = int(self.n_samples*val)

        dataset = {'train':{'x': self.x[:train_ind],
                            'y': self.y[:train_ind],
                            'l': self.l[:train_ind]},
                   'validation':{'x': self.x[train_ind:train_ind+val_ind],
                                 'y': self.y[train_ind:train_ind+val_ind],
                                 'l': self.l[train_ind:train_ind+val_ind]},
                   'test':{'x': self.x[train_ind+val_ind:],
                           'y': self.y[train_ind+val_ind:],
                           'l': self.l[train_ind+val_ind:]}
                  }
        return dataset

    def get_iterator(self, *args, batch_size=10):
        ds = tf.data.Dataset.from_tensor_slices(args)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(1)

        return ds
