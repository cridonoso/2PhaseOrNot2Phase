import tensorflow as tf 
from models.plstm import PhasedClassifier
from models.lstm import LSTMClassifier
import data
import numpy as np
import sys
import pickle
import os

def get_one_pred(dataset, rnn_unit, fold_n, norm):
    path_exp  = './experiments/{}_{}/fold_{}/{}_256/'.format(dataset, norm, fold_n, rnn_unit)
    path_data = './datasets/records/{}/fold_{}/{}/test.tfrecords'.format(dataset, fold_n, norm)

    # Loading testing data
    test_batches = data.load_record(path=path_data, batch_size=400)
    n_classes = [len(b[1][0]) for b in test_batches.take(1)][0]

    # Instancing Models
    name = '{}_{}/fold_{}/{}_{}'.format(dataset, 
                                        norm, 
                                        fold_n, 
                                        rnn_unit, 
                                        256)

    if rnn_unit == 'plstm':
        model = PhasedClassifier(256, 
                                 n_classes, 
                                 layers=2, 
                                 dropout=0.5, 
                                 lr=1e-3,
                                 name=name)
    else:
        model = LSTMClassifier(units=256, 
                               n_classes=n_classes, 
                               layers=2,
                               dropout=0.5,
                               lr=1e-3,
                               name=name)

    # Loading weights
    model.load_ckpt(path_exp+'/ckpts/')

    # Prediction
    y_pred, y_true = model.predict_proba(test_batches, concat_batches=True)

    return {fold_n: {rnn_unit : {'y_true': y_true, 'y_pred': y_pred}}}


dataset = sys.argv[1]
rnn_type = sys.argv[2]
fold_n = int(sys.argv[3])
norm = sys.argv[4]
print('dataset: {} - unit: {} - fold: {} - norm: {}'.format(dataset, rnn_type, fold_n, norm))

results = get_one_pred(dataset,  rnn_type, fold_n, norm)
path_to_save = './results/save/predictions/{}/fold_{}/'.format(dataset, fold_n)
os.makedirs(path_to_save, exist_ok=True)
with open('{}/{}_{}.pkl'.format(path_to_save, rnn_type, norm), 'wb') as h:
    pickle.dump(results, h)