from input import DataPreprocessing
from classifier import RNNClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

with open('data/linear.pkl', 'rb') as handle:
    d = pickle.load(handle)
    light_curves = d[0]
    labels = d[1]

data_object = DataPreprocessing(light_curves, labels=labels)
data_object.pad_series()
data_object.standardize()
dataset = data_object.train_test_val_split(val=0.25, test=0.25)

print(dataset['train']['x'].shape)
print(dataset['validation']['x'].shape)

train_iterator = data_object.get_iterator(dataset['train']['x'][...,1:],
                                          dataset['train']['x'][...,0],
                                          dataset['train']['y'],
                                          dataset['train']['l'],
                                          batch_size=100)

val_iterator = data_object.get_iterator(dataset['validation']['x'][...,1:],
                                        dataset['validation']['x'][...,0],
                                        dataset['validation']['y'],
                                        dataset['validation']['l'],
                                        batch_size=100)


n_classes = len(np.unique(data_object.y))
model = RNNClassifier(units=16, n_classes=n_classes)
model.fit(train_iterator, val_iterator, 5)
