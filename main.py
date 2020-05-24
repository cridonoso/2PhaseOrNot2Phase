from classifiers import PhasedClassifier, LSTMClassifier
import data
import tensorflow as tf


batch_size = 400
epochs     = 100 
units      = 16
fold_path  =  '../datasets/records/wise/fold_0/'
rnn_unit = 'lstm'


train_batches = data.load_record(path='{}/train.tfrecords'.format(fold_path), 
								batch_size=batch_size)
val_batches   = data.load_record(path='{}/val.tfrecords'.format(fold_path), 
                                 batch_size=batch_size)

n_classes = [len(b[1][0]) for b in train_batches.take(1)][0]

if rnn_unit == 'phased':
	model = PhasedClassifier(units=units, n_classes=n_classes, name='{}_{}'.format(rnn_unit, units))
if rnn_unit == 'lstm':
	model = LSTMClassifier(units=units, n_classes=n_classes, name='{}_{}'.format(rnn_unit, units))


model.fit(train_batches.take(2), 
		  val_batches.take(2), 
		  epochs, 
		  patience=10, 
		  save_path='./experiments/')
