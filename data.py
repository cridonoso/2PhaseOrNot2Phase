import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_example(lcid, label, lightcurve, mask, n_classes):
    f = dict()

    dict_features={
    'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(lcid).encode()])),
    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[lightcurve.shape[0]])),
    'n_classes':tf.train.Feature(int64_list=tf.train.Int64List(value=[n_classes]))
    }
    element_context = tf.train.Features(feature = dict_features)

    dict_sequence = {}
    for col in range(lightcurve.shape[1]):
        seqfeat = _float_feature(lightcurve[:, col])
        seqfeat = tf.train.FeatureList(feature = [seqfeat])
        dict_sequence['dim_{}'.format(col)] = seqfeat
    
    dict_sequence['mask'] = tf.train.FeatureList(feature = [_float_feature(mask)])
    
    element_lists = tf.train.FeatureLists(feature_list=dict_sequence)
    ex = tf.train.SequenceExample(context = element_context,
                                  feature_lists= element_lists)
    return ex


def create_record(light_curves, labels, masks, oids, path=''):
    ''' Create a tf record given a list of light curves

    Arguments:
        light_curves: list of light curves. 
                      Each sample should be a numpy array of n_obs x 3
                      where 3 = time, magnitude, error
        labels      : List of labels. Each class should be defined by an INT identifier
        path        : Path for saving
    '''

    if not path.endswith('.record'): path+='.record'
    
    n_classes = len(np.unique(labels))

    with tf.io.TFRecordWriter(path) as writer:
    
        for x, y, oid, mask in zip(light_curves, labels, oids, masks):

            # Make a record example for time series
            ex = get_example(oid, y, x, mask, n_classes) # number of classes whithin the dataset 
                                         # (needed for later onehot encoding)

            writer.write(ex.SerializeToString())

def get_sample(sample):
    '''
    Read a serialized sample and convert it to tensor
    '''
    context_features = {'label': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'length': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string),
                        'n_classes': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = dict()
    for i in range(3):
        sequence_features['dim_{}'.format(i)] = tf.io.VarLenFeature(dtype=tf.float32)
    
    sequence_features['mask'] = tf.io.VarLenFeature(dtype=tf.float32)
    
    context, sequence = tf.io.parse_single_sequence_example(
                            serialized=sample,
                            context_features=context_features,
                            sequence_features=sequence_features
                            )

    input_dict = dict()
    input_dict['lcid']   = tf.cast(context['id'], tf.string)
    input_dict['length'] = tf.cast(context['length'], tf.int32)
    input_dict['label']  = tf.cast(context['label'], tf.int32)

    casted_inp_parameters = []
    for i in range(3):
        seq_dim = sequence['dim_{}'.format(i)]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float32)
        casted_inp_parameters.append(seq_dim)

    sequence = tf.stack(casted_inp_parameters, axis=2)[0]
    
    mask = tf.sparse.to_dense(sequence['mask'])
    mask = tf.cast(mask, tf.bool)
            
    y = context['label']
    y_one_hot = tf.one_hot(y, tf.cast(context['n_classes'], tf.int32))
    
    return sequence, y_one_hot, mask

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
    dataset = dataset.map(lambda x: get_sample(x), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache
    dataset = dataset.cache() 
    batches = dataset.batch(batch_size)
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return batches
