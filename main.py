from models.lstm import LSTMClassifier
from models.plstm import PhasedClassifier
import data
import tensorflow as tf
import sys
from absl import app
from absl import flags
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "./data/records/train", "path")
flags.DEFINE_string("p", "ztf", "experiment name")
flags.DEFINE_string("rnn_unit", "plstm", "Recurrent unit (lstm or plstm)")
flags.DEFINE_integer("batch_size", 256, "number of samples involved in a single forward-backward")
flags.DEFINE_float('lr', 1e-3, "Learning rate")
flags.DEFINE_integer("epochs", 2000, "Number of epochs")
flags.DEFINE_integer("units", 256, "Number of neurons")
flags.DEFINE_integer("patience", 25, "Number of epochs to activate early stop")


def main(argv):
	train_batches = data.load_record(path='{}/train'.format(FLAGS.dataset),
									batch_size=FLAGS.batch_size,
									take=100)
	val_batches   = data.load_record(path='{}/val'.format(FLAGS.dataset),
	                                 batch_size=FLAGS.batch_size,
									 take=100)


	n_classes = [len(b[1][0]) for b in train_batches.take(1)][0]

	if FLAGS.rnn_unit == 'plstm':
		model = PhasedClassifier(units=FLAGS.units, n_classes=n_classes, name=FLAGS.p, lr=FLAGS.lr)
	if FLAGS.rnn_unit == 'lstm':
		model = LSTMClassifier(units=FLAGS.units, n_classes=n_classes, name=FLAGS.p, lr=FLAGS.lr)


	model.fit(train_batches,
			  val_batches,
			  FLAGS.epochs,
			  patience=FLAGS.patience,
			  save_path='./experiments/')

if __name__ == '__main__':
  app.run(main)
