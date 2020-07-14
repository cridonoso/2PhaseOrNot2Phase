import tensorflow as tf 
from tensorflow.keras.losses import categorical_crossentropy

class LDAM(tf.keras.losses.Loss):

	def __init__(self, cls_num, max_m, weight=None, s=10, 
				 reduction=tf.keras.losses.Reduction.NONE):
		''' LDAM Loss function
				
		Arguments:
			cls_num {[1D Tensor]} -- [Number of samples per class]
			max_m {[float]} -- [Maximum margin]
		
		Keyword Arguments:
			weight {[1D Tensor]} -- [Weights for losses] (default: {None})
			s {number} -- [scale factor] (default: {30})
		'''
		super(LDAM, self).__init__(reduction=reduction)
		cls_num = tf.divide(1., tf.sqrt(tf.sqrt(cls_num)))
		cls_num = cls_num * (max_m / tf.reduce_max(cls_num))

		self.margins = cls_num
		assert s > 0
		self.s = s
		self.weight = weight

	def call(self, y_true, y_pred):
		''' Calculates the LDAM loss based on the categorical xentropy
				
		Arguments:
			y_true {[int tensor]} -- [onehot matrix of labels]
			y_pred {[float tensor]} -- [predictions made by the classifier]
		
		Returns:
			[float tensor] -- [loss]
		'''

		batch_margin = tf.multiply(self.margins[None, :], y_true)
		batch_margin = tf.reduce_sum(batch_margin, -1)

		if len(y_pred.shape) == 2:
			y_m = y_pred - tf.expand_dims(batch_margin, 1)
		if len(y_pred.shape) == 3:
			y_m = y_pred - tf.expand_dims(batch_margin, 2)

		y_pred = tf.where(tf.cast(y_true, tf.bool), y_m, y_pred)

		cce_loss = categorical_crossentropy(y_true*self.s, y_pred)
		
		return cce_loss