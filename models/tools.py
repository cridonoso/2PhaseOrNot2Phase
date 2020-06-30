import tensorflow as tf 


def fix_tensor(x, mask):
    valid_mask = tf.cast(tf.reduce_sum(mask, 1), dtype=tf.bool)
    x = tf.boolean_mask(x, valid_mask)
    return x

def mask_pred(y_pred, mask):
    """ Get predictions according to the real lenght of the sample
    
    Changes the output probabilities of the padding for the 
    last valid prediction on each light curve
    
    Arguments:
        y_pred {[float tensor]} -- [tensor containing the output 
                                    probabilities for each time step]
        mask {[binary tensor]} -- [binary vector associated with each time step: 
                                   real (1) or padding (0)]
    
    Returns:
        [float tensor] -- [masked probabilities]
    """
    valid_mask = tf.cast(tf.reduce_sum(mask, 1), dtype=tf.bool)
    
    mask = tf.boolean_mask(mask, valid_mask)
    y_pred = tf.boolean_mask(y_pred, valid_mask)

    last = tf.cast(tf.reduce_sum(mask, 1), tf.int32)-1
    last = tf.stack([tf.range(last.shape[0]), last], 1)

    last_probas  = tf.gather_nd(y_pred, last)
    last_probas  = tf.expand_dims(last_probas, 2)
    tensor_last  = tf.tile(last_probas, [1,1,y_pred.shape[1]])
    inverse_mask = tf.cast(tf.where(mask == 0, 1, 0), dtype=tf.float32)
    tensor_last  = tf.transpose(tensor_last, [0,2,1]) * tf.expand_dims(inverse_mask, 2)
    tensor_first = y_pred * tf.expand_dims(mask, 2)
    return tensor_first + tensor_last

def add_scalar_log(value, writer, step, name):
    with writer.as_default():
        tf.summary.scalar(name, value, step=step)