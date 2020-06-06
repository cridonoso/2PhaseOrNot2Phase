import tensorflow as tf
import numpy as np


def read_tfrecord(serialized_example):
    context_features = {
        "label": tf.io.FixedLenFeature([], dtype=tf.int64),
        "n_classes": tf.io.FixedLenFeature([], dtype=tf.int64)
    }
    
    sequence_features = {
        "mjd": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "mags": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "err_mags": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "mask": tf.io.FixedLenSequenceFeature([], dtype=tf.float32)
    }

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features,
        context_features =context_features
    )
    
    x = tf.stack([sequence_parsed['mjd'],
                  sequence_parsed['mags'],
                  sequence_parsed['err_mags']
                  ], axis=1)
    
    y = context_parsed['label']
    y_one_hot = tf.one_hot(y, tf.cast(context_parsed['n_classes'], tf.int32))
    m = sequence_parsed['mask']
    
    return x, y_one_hot, m

def normalize(x, y, m):
    std_ = tf.expand_dims(tf.math.reduce_std(x, 1), 1)
    mean_ = tf.expand_dims(tf.math.reduce_mean(x, 1), 1)
    x = (x - mean_)/std_
    x = tf.where(tf.math.is_nan(x), 0., x)
    return x, y, m

def load_record(path, batch_size, standardize=False):
    """ Data loader for irregular time series with masking"
    
    Arguments:
        path {[str]} -- [record location]
        batch_size {[number]} -- [number of samples to be used in 
                                  neural forward pass]
    
    Returns:
        [tensorflow dataset] -- [batches to feed the model]
    """
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(lambda x: read_tfrecord(x), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if standardize:
        dataset = dataset.map(lambda x, y, m: normalize(x, y, m), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)    
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache
    dataset = dataset.cache() 
    batches = dataset.batch(batch_size)
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return batches
