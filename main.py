from models.lstm import LSTMClassifier
from models.plstm import PhasedClassifier
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
flags.DEFINE_string("rnn_unit", "lstm", "Recurrent unit (lstm or plstm)")
flags.DEFINE_integer("fold_n", 0, "Fold number whitin xvalidation.")
flags.DEFINE_integer("batch_size", 400, "number of samples involved in a single forward-backward")
flags.DEFINE_float('lr', 1e-3, "Learning rate")
flags.DEFINE_float('dropout', 0.5, "Dropout Probability to be applied at the output of each RNN")
flags.DEFINE_integer("epochs", 2000, "Number of epochs")
flags.DEFINE_integer("units", 256, "Number of neurons")
flags.DEFINE_integer("layers", 2, "Number of recurrent layers")
flags.DEFINE_integer("patience", 50, "Number of epochs to activate early stop")


def main(argv):
	fold_path  =  './datasets/records/{}/fold_{}/{}/'.format(FLAGS.dataset, 
															  FLAGS.fold_n, 
															  FLAGS.normalization)


	train_batches = data.load_record(path='{}/train.tfrecords'.format(fold_path), 
									batch_size=FLAGS.batch_size,
                                    shuffle=True)
	val_batches   = data.load_record(path='{}/val.tfrecords'.format(fold_path), 
	                                 batch_size=FLAGS.batch_size)


	n_classes = [len(b[1][0]) for b in train_batches.take(1)][0]
	name = '{}_{}/fold_{}/{}_{}'.format(FLAGS.dataset, 
										FLAGS.normalization, 
										FLAGS.fold_n, 
										FLAGS.rnn_unit, 
										FLAGS.units)

	if FLAGS.rnn_unit == 'plstm':
		model = PhasedClassifier(FLAGS.units, 
								 n_classes, 
								 layers=FLAGS.layers, 
								 dropout=FLAGS.dropout, 
								 lr=FLAGS.lr,
								 name=name)
	if FLAGS.rnn_unit == 'lstm':
		model = LSTMClassifier(units=FLAGS.units, 
							   n_classes=n_classes, 
							   layers=FLAGS.layers,
							   dropout=FLAGS.dropout,
							   lr=FLAGS.lr,
						   	   name=name)

	model.fit(train_batches, 
			  val_batches, 
			  FLAGS.epochs, 
			  patience=FLAGS.patience, 
			  save_path='./experiments/')

if __name__ == '__main__':
  app.run(main)
