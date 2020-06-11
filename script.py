#!/usr/bin/python
import sys
import subprocess
import time

# dataset = sus.argv[1]
# unit_type = sys.argv[2]

for dataset in ['linear', 'wise', 'css']:
	for unit_type in ['phased', 'lstm']:
		for normalization in ['n2']:
			for fold_n in [0,1,2]:
			    start = time. time()
			    command1 = 'python main.py {} {} {} {}'.format(dataset, fold_n, unit_type, normalization)
			    print('executing: ',command1)
			    try:
			        subprocess.call(command1, shell=True)
			    except:
			        print('ERROR IN: ',command1)
			    end = time. time()
			    print('{} fold_{} takes {}'.format(unit_type, fold_n,(end - start)))