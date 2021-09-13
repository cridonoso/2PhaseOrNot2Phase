#!/usr/bin/python
import sys
import subprocess
import time


dataset = sys.argv[1]

for unit_type in ['plstm', 'lstm']:
	start = time. time()
	command1 = 'python main.py --dataset {} \
							   --rnn_unit {} \
							   --p {}_{} \
							   --take 300'.format(dataset,
						    			     	  unit_type,
												  dataset, unit_type)
	try:
	    subprocess.call(command1, shell=True)
	except Exception as e:
	    print(e)

	end = time. time()
	print('{} fold_{} takes {}'.format(unit_type, fold_n,(end - start)))
