from classifiers import PhasedClassifier, LSTMClassifier
import data
import tensorflow as tf
import sys

batch_size = 400
epochs     = 100 
units      = 256

dataset   = sys.argv[1]
fold_n    = sys.argv[2]
rnn_unit = sys.argv[3]

fold_path  =  '../datasets/records/{}/fold_{}/'.format(dataset, fold_n)


train_batches = data.load_record(path='{}/train.tfrecords'.format(fold_path), 
								batch_size=batch_size)
val_batches   = data.load_record(path='{}/val.tfrecords'.format(fold_path), 
                                 batch_size=batch_size)

n_classes = [len(b[1][0]) for b in train_batches.take(1)][0]

if rnn_unit == 'phased':
	model = PhasedClassifier(units=units, n_classes=n_classes, name='{}/fold_{}/{}_{}'.format(dataset, fold_n, rnn_unit, units), normalize=True)
if rnn_unit == 'lstm':
	model = LSTMClassifier(units=units, n_classes=n_classes, name='{}/fold_{}/{}_{}'.format(dataset, fold_n, rnn_unit, units))


model.fit(train_batches, 
		  val_batches, 
		  epochs, 
		  patience=10, 
		  save_path='./experiments/')
