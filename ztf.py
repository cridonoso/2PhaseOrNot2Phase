import numpy as np
import pandas as pd

class_code = {
			'AGN'	 :0,
			'Blazar' :1, 
			'CV/Nova':2,
			'Ceph'	 :3,
			'DSCT'	 :4,
			'EA'	 :5,
			'EB/EW'	 :6,
			'LPV'	 :7,
			'NLAGN'	 :8,
			'NLQSO'	 :9,
			'Periodic-Other':10,
			'QSO'	 :11,
			'RRL'	 :12,
			'RSCVn'	 :13,
			'SLSN'	 :14,
			'SNII'	 :14,
			'SNIIb'	 :14,
			'SNIIn'	 :14,
			'SNIa'	 :14,
			'SNIbc'	 :14,
			'TDE'	 :14,
			'YSO'	 :15,
			'ZZ'	 :16
			}

def get_light_curves(metapath, det_path, nondet_path='', chunks=True, chunksize=1e6):
	"""Open a csv of ZTF detection and convert observation 
	to a list of lightcurves
	
	Arguments:
		metapath {[str]} -- [metadata associated with objects 
							(e.g., class)]
		det_path {[str]} -- [csv with detections]
	
	Keyword Arguments:
		nondet_path {str} -- [csv with nondetection] (default: {''})
	
	Returns:
		[type] -- [description]
	"""

	light_curves = []
	labels 		 = []
	oids 		 = []
	metadata_df  = pd.read_csv(metapath)

	if chunks:
		for chunk in pd.read_csv(det_path, chunksize=chunksize, low_memory=False):
			result = pd.merge(chunk[['oid', 'mjd', 'magpsf_corr', 'sigmapsf_corr', 'fid']], 
							  metadata_df[['oid', 'classALeRCE']], 
							  on='oid')

			objects = result.groupby('oid')
			
			for object_id, serie in objects:
				lc = serie.iloc[:, 1:-1]
				label = serie.iloc[0, -1]
				light_curves.append(lc)
				labels.append(class_code[label])
				oids.append(object_id)
	else:
		# ======== RAM expensive ==========
		detections = pd.read_csv(det_path, low_memory=False)
		result = pd.merge(detections[['oid', 'mjd', 'magpsf_corr', 'sigmapsf_corr', 'fid']], 
				  		  metadata_df[['oid', 'classALeRCE']], 
				  		  on='oid')

		objects = result.groupby('oid')
		for object_id, serie in objects:
			lc = serie.iloc[:, 1:-1]
			label = serie.iloc[0, -1]
			light_curves.append(lc.values)
			labels.append(class_code[label])
			oids.append(object_id)

	return light_curves, labels, oids


def pad_lightcurves(lightcurves, labels, maxobs=200):
	''' Take a list of lightcurves and pad them. 

	It also takes subsamples of observations based on 
	maxobs. In other words, we cut the light curve
	in batches of <maxobs> observations

	Arguments:
		lightcurves {[list]} -- [list of light curves]
		labels {[labels]} -- [list of label codes]

	Keyword Arguments:
		maxobs {number} -- [max number of observations whitin 
							the light curve] (default: {200})

	Returns:
		[numpy array] -- [padded dataset with their corresponding masks]
	'''
	n_samples = len(lightcurves)

	new_lightcurves = []
	new_labels = []
	masks = []
	for k in range(n_samples):
		# === Check if times are sorted ====
		indices = np.argsort(lightcurves[k][...,0])
		lc = lightcurves[k][indices]

		# === Split Light Curve ============
		n_div = lc.shape[0]%maxobs
		base_lc = np.zeros(shape=[lc.shape[0]+(maxobs-n_div), lc.shape[-1]])
		base_lc[:lc.shape[0], :] = lc
		# === Get mask =====================
		base_mask = np.zeros(shape=[lc.shape[0]+(maxobs-n_div)])
		base_mask[:lc.shape[0]] = np.ones(lc.shape[0])
		# Split matrix and write record
		splits_lc = np.split(base_lc, int(base_lc.shape[0]/maxobs))
		splits_mask = np.split(base_mask, int(base_mask.shape[0]/maxobs))

		for s in range(len(splits_lc)):
			new_lightcurves.append(splits_lc[s])
			new_labels.append(labels[k])
			masks.append(splits_mask[s])

	return np.array(new_lightcurves), np.array(new_labels), np.array(masks)

def train_val_test_split(lightcurves, labels, train_frac=0.5, val_frac=0.25):
	''' Divide dataset in training, validation and testing set.


	Arguments:
		lightcurves {[list]} -- [list of light curves]
		labels {[list]} -- [list of labels]

	Keyword Arguments:
		train_frac {float} -- [sample fraction for training] (default: {0.5})
		val_frac {float} -- [sample fraction for validation] (default: {0.25})
	
	Returns:
		[dictonary] -- [dict with 'train' - 'validation' - 'test' subsets]
	'''

	X_train = []
	X_valid = []
	X_test  = []

	y_train = []
	y_valid = []
	y_test  = []


	uniques, counts = np.unique(labels, return_counts=True)
	for u, c in zip(uniques, counts):
		particular_class = np.array(lightcurves)[labels == u]
		n = particular_class.shape[0]

		indices = np.arange(0, n)
		np.random.shuffle(indices)
		particular_class = particular_class[indices]

		ntrain = int(np.floor(n*train_frac))
		nval   = int(np.floor(n*val_frac))

		X_train.append(particular_class[:ntrain])
		y_train.append(np.tile(u, ntrain))

		X_valid.append(particular_class[ntrain: ntrain+nval])
		y_valid.append(np.tile(u, nval))

		X_test.append(particular_class[(ntrain+nval):])
		y_test.append(np.tile(u, n-(ntrain+nval)))

	X_train = np.concatenate(X_train)
	X_valid = np.concatenate(X_valid)
	X_test = np.concatenate(X_test)

	y_train = np.concatenate(y_train)
	y_valid = np.concatenate(y_valid)
	y_test = np.concatenate(y_test)

	return {'train': {'x': X_train, 'y': y_train},
			'validation': {'x': X_valid, 'y': y_valid},
			'test': {'x': X_test, 'y': y_test}}

def sanity_check(lightcurves):
	'''Operation for cleaning lightcurves
	
	Arguments:
		lightcurves {[list]} -- [list of lightcurves]
	'''

	# Filtering NaNs

	# Sorting by MJD

	# 2 bands inclusion

	return  lightcurves

def gp_subset(lightcurves, minobs=10):
	""" Apply some criterias for selecting good 
	lightcurves to adjust the GP. 

	Arguments:
		lightcurves {[list]} -- [list of light curves]
	
	Keyword Arguments:
		minobs {number} -- [Minimum observation for adjusting GP] 
						   (default: {10})
	
	Returns:
		[list] -- [list of observations]
	"""
	subset = []

	# Filtering by numer of observations

	# Compare times. The bands are expected to have close 
	# (in terms of mjd) observations.

	return subset

def train_gp(lightcurves, mode='periodic'):
	''' Fit GP mdoels for each light curve whitin the dataaset
	
	Kernels should be different between types of stars
	
	Arguments:
		lightcurves {[list]} -- [light curves]
	
	Keyword Arguments:
		mode {str} -- [Object type 
						- use 'periodic', 'stochastic', 'transient'] 
					  (default: {'periodic'})
	Returns:
		[type] -- [description]
	'''
	models = []

	if mode == 'periodic':
		kernel = None
		pass

	if mode == 'stochastic':
		kernel = None
		pass

	if mode == 'transient':
		kernel = None
		pass

	return models


def sample_lightcurves(models, n_samples, obs=200):
	'''Sample light curves from adjusted GP
	
	Arguments:
		models {[list]} -- [list of trained models]
		n_samples {[type]} -- [number of samples to augment]
	
	Keyword Arguments:
		obs {number} -- [description] (default: {200})
	'''

	return samples 

