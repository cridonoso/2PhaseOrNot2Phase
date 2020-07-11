#!/usr/bin/python
import sys
import subprocess
import time


dataset = sys.argv[1]

for unit_type in ['plstm', 'lstm']:
	for normalization in ['n1','n2']:
		for fold_n in range(3):
		    start = time. time()
		    command1 = 'python main.py --dataset {} \
		    						   --fold_n  {} \
		    						   --rnn_unit {} \
		    						   --normalization {}'.format(dataset, 
		    													  fold_n, 
		    													  unit_type, 
		    													  normalization)
		    print('::: TRAINING SCRIPT :::\n:::::::::::::::::::::::\n- dataset: {}\n- unit: {}\n- norm: {}\n- fold: {}\n:::::::::::::::::::::::'.format(dataset,
		    																																			unit_type,
		    																																 			normalization,
		    																																 			fold_n),
		    	  end='\r')
		    try:
		        subprocess.call(command1, shell=True)
		    except Exception as e:
		        print(e)
		    end = time. time()
		    print('{} fold_{} takes {}'.format(unit_type, fold_n,(end - start)))