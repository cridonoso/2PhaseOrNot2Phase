import numpy as np
import pandas as pd

class_code = {
            'AGN'	 :0,
            'Blazar' :1, 
            'CV/Nova':2,
            'Ceph'	 :3,
            'DSCT'	 :4,
            'EA'	 :5,
            'EB/EW'	 :5,
            'LPV'	 :6,
            'NLAGN'	 :-1,
            'NLQSO'	 :-1,
            'Periodic-Other':7,
            'QSO'	 :8,
            'RRL'	 :9,
            'RSCVn'	 :-1,
            'SLSN'	 :10,
            'SNII'	 :11,
            'SNIIb'	 :11,
            'SNIIn'	 :11,
            'SNIa'	 :12,
            'SNIbc'	 :13,
            'TDE'	 :-1,
            'YSO'	 :14,
            'ZZ'	 :-1
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
        result = pd.merge(detections[['oid', 'mjd', 'magpsf_corr', 'sigmapsf_corr', 'fid', 'rb', 'corrected']], 
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

def pad_lightcurves(lightcurves, labels, oids, maxobs=200):
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
    new_oids = []
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
            new_oids.append(oids[k])

    return np.array(new_lightcurves), np.array(new_labels), np.array(masks), np.array(new_oids)

