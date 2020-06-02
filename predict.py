import tensorflow as tf 
from classifiers import PhasedClassifier, LSTMClassifier
import data
import numpy as np
import sys
import pickle


def get_one_pred(dataset, rnn_unit, fold_n):
    path_exp  = './experiments/{}/fold_{}/{}_256/'.format(dataset, fold_n, rnn_unit)
    path_data = '../datasets/records/{}/fold_{}/test.tfrecords'.format(dataset, fold_n)

    # Loading testing data
    test_batches = data.load_record(path=path_data, batch_size=500)
    n_classes = [len(b[1][0]) for b in test_batches.take(1)][0]

    # Instancing Models
    if rnn_unit == 'phased':
        model = PhasedClassifier(units=256, 
                                 n_classes=n_classes,
                                 use_old=False)
    else:
        model = LSTMClassifier(units=256, 
                               n_classes=n_classes)

    # Loading weights
    model.load_ckpt(path_exp+'/ckpts/')

    # Prediction
    y_pred, y_true = model.predict_proba(test_batches)

    return {fold_n: {rnn_unit : {'y_true': y_true, 'y_pred': y_pred}}}

dataset = sys.argv[1]
rnn_type = sys.argv[2]
fold_n = int(sys.argv[3])
print('dataset: {} - units: {} - fold: {}'.format(dataset, rnn_type, fold_n))
results = get_one_pred(dataset, 'lstm', fold_n)

with open('./predictions/{}_{}_{}.pkl'.format(dataset, rnn_type, fold_n), 'wb') as handle:
	pickle.dump(results, handle)