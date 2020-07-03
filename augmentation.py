import numpy as np


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

	return light_curves


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