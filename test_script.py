#!/usr/bin/python
import sys
import subprocess
import time

dataset  = sys.argv[1]

for norm in ['n1', 'n2']:
    for unit_type in ['lstm', 'plstm']:
        for fold_n in range(3):
            start = time. time()
            command1 = 'python predict.py {} {} {} {}'.format(dataset, 
                                                              unit_type, 
                                                              fold_n,
                                                              norm)
            print('executing: ',command1)
            try:
                subprocess.call(command1, shell=True)
            except:
                print('ERROR IN: ',command1)
            end = time. time()
            print('{} fold_{} takes {}'.format(unit_type, fold_n,(end - start)))