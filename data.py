import tensorflow as tf
import numpy as np


def create_record(light_curves, labels, path=''):
    ''' Create a tf record given a list of light curves

    Arguments:
        light_curves: list of light curves. 
                      Each sample should be a numpy array of n_obs x 3
                      where 3 = time, magnitude, error
        labels      : List of labels. Each class should be defined by an INT identifier
        path        : Path for saving
    '''

    if not path.endswith('.tfrecords'): path+='.tfrecords'

    n_samples = len(light_curves)
    max_obs   = np.max([x.shape[0] for x in light_curves])
    n_classes = len(np.unique(labels))

    with tf.io.TFRecordWriter(path) as writer:
    
        for x, y in zip(light_curves, labels):

            n_obs = x.shape[0] # current number of observations
            mask = np.zeros(shape=[max_obs]) # first create an empy vector
            mask[:n_obs] = np.ones(n_obs) # then replace by ones according to the criteria


            # Make a record example for time series
            ex = make_example(x[:,0], # time
                              x[:,1], # mean magnitude
                              x[:,2], # std magnitude
                              mask, 
                              y, # integer related with the object class 
                              n_classes) # number of classes whithin the dataset 
                                         # (needed for later onehot encoding)

            writer.write(ex.SerializeToString())

def make_example(mjd, mags, err_mags, masks, label, n_classes):
    ''' Create a record example 

    Given a list of times, mag and errors for a particular light curve 
    create an example to be saved in tf.record binary format

    Arguments:
        mjd {[numpy array]} -- [array of times (e.g., MJD)]
        mags {[numpy array]} -- [array of magnitudes (it could be flux too)]
        err_mags {[numpy array]} -- [array of observational errors]
        mask {[numpy array]} -- 
    Returns:
        [record example] -- [record sequence example]
    '''

    ex = tf.train.SequenceExample()
    # No depends on time
    ex.context.feature["label"].int64_list.value.append(np.int32(label))
    ex.context.feature["n_classes"].int64_list.value.append(np.int32(n_classes))

    # Temporal features
    fl_mjd = ex.feature_lists.feature_list["mjd"]
    fl_flux = ex.feature_lists.feature_list["mags"]
    fl_err = ex.feature_lists.feature_list["err_mags"]
    fl_mask = ex.feature_lists.feature_list["mask"]

    for index in tf.range(len(mjd)):
        fl_mjd.feature.add().float_list.value.append(mjd[index])
        fl_flux.feature.add().float_list.value.append(mags[index])
        fl_err.feature.add().float_list.value.append(err_mags[index])
        fl_mask.feature.add().float_list.value.append(masks[index])

    return ex

def read_tfrecord(serialized_example):
    """Read record serialized example
    
    Arguments:
        serialized_example {[string record]} -- [example containing the light curve]
    
    Returns:
        [x3 tensors] -- [Tensor relating with the observations, one hot labels and mask]
    """
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
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache
    dataset = dataset.cache() 
    batches = dataset.batch(batch_size)
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return batches
