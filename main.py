from classifier import RNNClassifier
import data


batch_size = 100
epochs     = 5 
fold_path  =  '../datasets/records/linear/fold_0/'

train_batches = data.load_record(path='{}/train.tfrecords'.format(fold_path), 
                                 batch_size=batch_size)
val_batches   = data.load_record(path='{}/val.tfrecords'.format(fold_path), 
                                 batch_size=batch_size)


model = RNNClassifier(units=16)
model.fit(train_batches, 
          val_batches, 
          epochs, 
          patience=5, 
          save_path='./experiments/')
