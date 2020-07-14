from models.lstm import LSTMClassifier
from models.plstm import PhasedClassifier
from models.imlstm import ImbalancedLSTM
import data
import tensorflow as tf
import sys
from absl import app
from absl import flags
import os
try:
	os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
except:
	os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "linear", "dataset (linear - macho - ogle - asas - css - gaia - wise)")
flags.DEFINE_string("normalization", "n1", "Normalization approach according to the preprocessing step (n1 or n2)")
flags.DEFINE_string("rnn_unit", "imlstm", "Recurrent unit (lstm or plstm)")
flags.DEFINE_integer("fold_n", 0, "Fold number whitin xvalidation.")
flags.DEFINE_integer("batch_size", 400, "number of samples involved in a single forward-backward")
flags.DEFINE_float('lr', 1e-3, "Learning rate")
flags.DEFINE_float('dropout', 0.5, "Dropout probability applied over the output of the RNN")
flags.DEFINE_integer("epochs", 2000, "Number of epochs")
flags.DEFINE_integer("units", 32, "Number of neurons")
flags.DEFINE_integer("layers", 2, "Number of layers")
flags.DEFINE_integer("patience", 30, "Number of epochs to activate early stop")


def get_cls_num(batches, n_classes):
	cls_num = tf.zeros(n_classes)
	for batch in batches:
		cls_num += tf.reduce_sum(batch[1], 0)
	return cls_num

def main(argv):
	fold_path  =  './datasets/records/{}/fold_{}/{}/'.format(FLAGS.dataset, 
															  FLAGS.fold_n, 
															  FLAGS.normalization)


	train_batches = data.load_record(path='{}/train.tfrecords'.format(fold_path), 
									batch_size=FLAGS.batch_size)
	val_batches   = data.load_record(path='{}/val.tfrecords'.format(fold_path), 
	                                 batch_size=FLAGS.batch_size)


	n_classes = [len(b[1][0]) for b in train_batches.take(1)][0]

	name = '{}_{}/fold_{}/{}_{}'.format(FLAGS.dataset, 
										FLAGS.normalization, 
										FLAGS.fold_n, 
										FLAGS.rnn_unit, 
										FLAGS.units)

	if FLAGS.rnn_unit == 'plstm':
		model = PhasedClassifier(units=FLAGS.units, n_classes=n_classes, name=name, lr=FLAGS.lr)
	if FLAGS.rnn_unit == 'lstm':
		model = LSTMClassifier(units=FLAGS.units, n_classes=n_classes, name=name, lr=FLAGS.lr)
	if FLAGS.rnn_unit == 'imlstm':
		cls_num = get_cls_num(train_batches, n_classes)
		model = ImbalancedLSTM(units=FLAGS.units, cls_num=cls_num, layers=FLAGS.layers, 
							   max_m=0.5, dropout=FLAGS.dropout, name=name, lr=FLAGS.lr)

	model.fit(train_batches, 
			  val_batches, 
			  FLAGS.epochs, 
			  patience=FLAGS.patience, 
			  save_path='./experiments/')

if __name__ == '__main__':
  app.run(main)
