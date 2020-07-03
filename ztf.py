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

def get_light_curves(metapath, det_path, nondet_path=''):
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

	for chunk in pd.read_csv(det_path, chunksize=1e6, low_memory=False):
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

	return light_curves, labels, oids


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
