from classifiers import PhasedClassifier, LSTMClassifier
import data
import tensorflow as tf


batch_size = 100
epochs     = 5 
fold_path  =  '../datasets/records/wise/fold_0/'
rnn_unit = 'phased'


train_batches = data.load_record(path='{}/train.tfrecords'.format(fold_path), 
                                 batch_size=batch_size)
val_batches   = data.load_record(path='{}/val.tfrecords'.format(fold_path), 
                                 batch_size=batch_size)

n_classes = [len(b[1][0]) for b in train_batches.take(1)][0]

if rnn_unit == 'phased':
	model = PhasedClassifier(units=128, n_classes=n_classes)
if rnn_unit == 'lstm':
	model = LSTMClassifier(units=128, n_classes=n_classes)


model.fit(train_batches, 
          val_batches, 
          epochs, 
          patience=10, 
          save_path='./experiments/')
