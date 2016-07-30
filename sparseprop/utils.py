import numpy as np
import pandas as pd

from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import MeanShift

from numpy.lib.stride_tricks import as_strided as ast

def get_typical_durations(raw_durations, bandwidth_percentile=0.05, 
                       min_intersection=0.5, miss_covered=0.1):
    """Return typical durations in a dataset."""
    dur = (raw_durations).reshape(raw_durations.shape[0], 1)
    bandwidth = estimate_bandwidth(dur, quantile=bandwidth_percentile)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
    ms.fit(dur.reshape((dur.shape[0]), 1))
    tw = np.sort(np.array(
        ms.cluster_centers_.reshape(ms.cluster_centers_.shape[0]), dtype=int))
    # Warranty a min intersection in the output durations.
    p = np.zeros((dur.shape[0], tw.shape[0]))
    for idx in range(tw.shape[0]):
        p[:, idx] = (dur/tw[idx]).reshape(p[:,idx].shape[0])
    ll = (p>=min_intersection) & (p<=1.0/min_intersection)
    if (ll.sum(axis=1)>0).sum() / float(df.shape[0]) < (1.0-miss_covered):
        assert False, "Condition of minimum intersection not satisfied"
    return tw

def wrapper_nms(proposal_df, overlap=0.65):
    """Apply non-max-suppresion to a video batch.
    """
    vds_unique = pd.unique(proposal_df['video-name'])
    new_proposal_df = []
    for i, v in enumerate(vds_unique):
        idx = proposal_df['video-name'] == v
        p = proposal_df.loc[idx, ['video-name', 'f-init', 'f-end',
                                  'score', 'video-frames']]
        n_frames = np.int(p['video-frames'].mean())
        loc = np.stack((p['f-init'], p['f-end']), axis=-1)
        loc, score = nms_detections(loc, np.array(p['score']), overlap)
        n_proposals = score.shape[0]
        n_frames = np.repeat(p['video-frames'].mean(), n_proposals).astype(int)
        this_df = pd.DataFrame({'video-name': np.repeat(v, n_proposals),
                                'f-init': loc[:, 0], 'f-end': loc[:, 1],
                                'score': score,
                                'video-frames': n_frames})
        new_proposal_df.append(this_df)
    return pd.concat(new_proposal_df, axis=0)

def nms_detections(dets, score, overlap=0.65):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.
    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.
    Parameters
    ----------
    dets : ndarray.
        Each row is ['f-init', 'f-end']
    score : 1darray.
        Detection score.
    overlap : float.
        Minimum overlap ratio (0.3 default).
    Outputs
    -------
    dets : ndarray.
        Remaining after suppression.
    """
    t1 = dets[:, 0]
    t2 = dets[:, 1]
    ind = np.argsort(score)

    area = (t2 - t1 + 1).astype(float)

    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]

        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])

        wh = np.maximum(0., tt2 - tt1 + 1.0)
        o = wh / (area[i] + area[ind] - wh)

        ind = ind[np.nonzero(o <= overlap)[0]]

    return dets[pick, :], score[pick]
