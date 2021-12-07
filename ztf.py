import numpy as np
import pandas as pd


def get_light_curves(metadata_df, detections, class_code, n_min=1):
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

    result = pd.merge(detections,
                      metadata_df[['oid', 'alerceclass', 'partition']],
                      on='oid')

    objects = result.groupby('oid')
    for object_id, serie in objects:
        lc = serie.iloc[:, 1:-2]
        label = serie.iloc[0, -2]
        if lc.shape[0] >= n_min:
            light_curves.append(lc.values)
            labels.append(class_code.index(label))
            oids.append(object_id)

    return np.array(light_curves), np.array(labels), np.array(oids, dtype='object')

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
            if np.sum(splits_mask[s]) == 0:
                continue
            new_lightcurves.append(splits_lc[s])
            new_labels.append(labels[k])
            masks.append(splits_mask[s])
            new_oids.append(oids[k])

    return np.array(new_lightcurves), np.array(new_labels), np.array(masks), np.array(new_oids)
