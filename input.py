import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import math

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

    def pad_series(self, value=0, max_obs=-1):
        self.x = pad_sequences(sequences=self.x,
                               dtype='float32',
                               padding='post',
                               value=value)

    def get_mask(self, lens):
        print('[INFO] Processing mask, please wait ...')
        mask_list = tf.TensorArray(tf.float32, size=lens.shape[0])
        for index in tf.range(lens.shape[0]):
            valid = tf.ones(lens[index], dtype=tf.float32)
            no_valid = tf.zeros(self.maxlen-lens[index], dtype=tf.float32)
            mask = tf.concat([valid, no_valid], axis=0)
            mask_list.write(index, mask)
        return mask_list.stack()

    def split_series(self, x, y, lens_list, max_obs):
        offset = max_obs - (self.maxlen % max_obs)
        to_add = tf.zeros([x.shape[0], offset, x.shape[2]])
        x_extended = tf.concat([x, to_add], axis=1)

        div_n = int((self.maxlen+offset)/max_obs)
        divisions = tf.split(x_extended, div_n, axis=1)
        labels = np.array([y[x] for x in range(len(x)) for _ in range(div_n)])
        new_x = tf.concat(divisions, axis=0)

        boolean_mask = tf.cast(tf.reduce_sum(new_x, axis=[1,2]), dtype=tf.bool)
        no_zeros = tf.boolean_mask(new_x, boolean_mask, axis=0)
        labels_no_zeros = tf.boolean_mask(labels, boolean_mask, axis=0)

        # lens
        lens_new = []
        ids = []
        for _ in range(div_n):
            for k, l in enumerate(lens_list):
                if l < max_obs:
                    lens_new.append(l)
                else:
                    lens_new.append(max_obs)
                    l-=max_obs
                ids.append(k)
        lens_new_no_zeros = tf.boolean_mask(lens_new, boolean_mask, axis=0)
        ids_no_zeros = tf.boolean_mask(ids, boolean_mask, axis=0)
        return no_zeros, labels_no_zeros, lens_new_no_zeros, ids_no_zeros

    def train_test_val_split(self, val=0.25, test=0.25, max_obs=-1):
        indices = np.arange(0, self.n_samples)
        np.random.shuffle(indices)

        self.x = self.x[indices]
        self.y = self.y[indices]
        self.l = self.l[indices]

        train_ind = int(self.n_samples*(1-(val+test)))
        val_ind   = int(self.n_samples*val)

        if max_obs == -1:
            dataset = {'train':{'x': self.x[:train_ind],
                                'y': self.y[:train_ind],
                                'l': self.l[:train_ind],
                                'i': indices[:train_ind]},
                       'validation':{'x': self.x[train_ind:train_ind+val_ind],
                                     'y': self.y[train_ind:train_ind+val_ind],
                                     'l': self.l[train_ind:train_ind+val_ind],
                                     'i': indices[train_ind:train_ind+val_ind]},
                       'test':{'x': self.x[train_ind+val_ind:],
                               'y': self.y[train_ind+val_ind:],
                               'l': self.l[train_ind+val_ind:],
                               'i': indices[train_ind+val_ind:]}
                      }
            return dataset
        else:
            train_x, train_y, \
            train_l, train_i = self.split_series(self.x[:train_ind],
                                                 self.y[:train_ind],
                                                 self.l[:train_ind],
                                                 max_obs)
            val_x, val_y, \
            val_l, val_i = self.split_series(self.x[train_ind:train_ind+val_ind],
                                             self.y[train_ind:train_ind+val_ind],
                                             self.l[train_ind:train_ind+val_ind],
                                             max_obs)
            test_x, test_y, \
            test_l, test_i = self.split_series(self.x[train_ind+val_ind:],
                                               self.y[train_ind+val_ind:],
                                               self.l[train_ind+val_ind:],
                                               max_obs)
            self.maxlen = max_obs

            return {'train':{'x': train_x,
                             'y': train_y,
                             'l': self.get_mask(train_l),
                             'i': train_i},
                    'validation':{'x': val_x,
                                  'y': val_y,
                                  'l': self.get_mask(val_l),
                                  'i': val_i},
                    'test':{'x': test_x,
                            'y': test_y,
                            'l': self.get_mask(test_l),
                            'i': test_i}}

    def get_iterator(self, *args, batch_size=10):
        ds = tf.data.Dataset.from_tensor_slices(args)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(1)

        return ds
