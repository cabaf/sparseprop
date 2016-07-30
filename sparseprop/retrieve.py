import numpy as np
import pandas as pd

from joblib import delayed
from sklearn.preprocessing import normalize
from joblib import Parallel

import spams

from feature import C3D as FeatHelper

""" 
This interface allows to retrieve action proposals using a Class-Indepedent 
or Class-Induced models - Caba et al. CVPR 2016.
"""

def generate_candidate_proposals(video_info, proposal_sizes,
                                 feat_size=16, stride_intersection=0.1):
    """Returns a set of candidate proposals for a given video.
    
    Parameters
    ----------
    video_info : DataFrame
        DataFrame containing the 'video-name' and 'video-frames'.
    proposal_sizes : 1darray
        np array containing list of proposal sizes.
    feat_size : int, optional
        Size of the temporal extension of the features.
    stride_intersection : float, optional
        Percentage of intersection between temporal windows.

    Outputs
    -------
    proposal_df : DataFrame
        DataFrame containing the candidate proposals. It is 
        formatted as follows: 'video-name', 'f-init', 'n-frames'.
    """
    proposal_lst = []
    a = None
    # Sanitize
    video_info['video-frames'] = int(video_info['video-frames'])
    for p_size in proposal_sizes:
        if (video_info['video-frames'] - feat_size) < p_size:
            continue
        step_size = int(p_size * stride_intersection)
        # Sliding windows
        this_proposals = np.arange(
            0, video_info['video-frames'] - p_size - feat_size, step_size)
        this_proposals = np.vstack((this_proposals,
                                    np.repeat(p_size, 
                                              this_proposals.shape[0])))
        proposal_lst.append(this_proposals)
    # If video is too small and not proposals were generated.
    if not proposal_lst:
        return
    proposal_stack = np.hstack(proposal_lst).T
    n_proposals = proposal_stack.shape[0]
    proposal_df = pd.DataFrame({'video-name': np.repeat(
                                    video_info['video-name'], 
                                    n_proposals),
                                'f-init': proposal_stack[:, 0],
                                'n-frames': proposal_stack[:, 1],
                                'video-frames': np.repeat(
                                    video_info['video-frames'],
                                    n_proposals),
                                'score': np.zeros(n_proposals)})
    return proposal_df

def retrieve_proposals(video_info, model, feature_filename,
                       feat_size=16, stride_intersection=0.1):
    """Retrieve proposals for a given video.
    
    Parameters
    ----------
    video_info : DataFrame
        DataFrame containing the 'video-name' and 'video-frames'.
    model : dict
        Dictionary containing the learned model.
        Keys: 
            'D': 2darray containing the sparse dictionary.
            'cost': Cost function at the last iteration.
            'durations': 1darray containing typical durations (n-frames)
                 in the training set.
            'type': Dictionary type.
    feature_filename : str
        String containing the path to the HDF5 file containing 
        the features for each video. The HDF5 file must contain 
        a group for each video where the id of the group is the name 
        of the video; and each group must contain a dataset containing
        the features.
    feat_size : int, optional
        Size of the temporal extension of the features.
    stride_intersection : float, optional
         Percentage of intersection between temporal windows.
    """
    feat_obj = FeatHelper(feature_filename, t_stride=1)
    candidate_df = generate_candidate_proposals(video_info, model['durations'],
                                                feat_size, stride_intersection)
    D = model['D']
    params = model['params']
    feat_obj.open_instance()
    feat_stack = feat_obj.read_feat(video_info['video-name'])
    feat_obj.close_instance()
    n_feats = feat_stack.shape[0]
    candidate_df = candidate_df[
        (candidate_df['f-init'] + candidate_df['n-frames']) <= n_feats]
    candidate_df = candidate_df.reset_index(drop=True)
    proposal_df = Parallel(n_jobs=-1)(delayed(wrapper_score_proposals)(this_df,
                                                                      D, 
                                                                     feat_stack,
                                                                       params,
                                                                     feat_size)
                                      for k, this_df in candidate_df.iterrows())
    proposal_df = pd.concat(proposal_df, axis=1).T
    proposal_df['score'] = (
        proposal_df['score'] - proposal_df['score'].min()) / (
            proposal_df['score'].max() - proposal_df['score'].min())
    proposal_df['score'] = np.abs(proposal_df['score'] - 1.0)
    proposal_df = proposal_df.loc[proposal_df['score'].argsort()[::-1]]
    proposal_df = proposal_df.rename(columns={'n-frames': 'f-end'})
    proposal_df['f-end'] = proposal_df['f-init'] + proposal_df['f-end'] - 1
    return proposal_df.reset_index(drop=True)

def score_proposals(X, D, params):
    """ Scores a proposal segment using the reconstruction error 
        from a pretrained dictionary.
    """
    X = np.asfortranarray(X.T.copy())
    D = np.asfortranarray(D.T.copy())
    A_0 = np.zeros((D.shape[1], X.shape[1]), order='FORTRAN')
    A = spams.fistaFlat(X, D, A_0, **params)
    cost = (1.0/X.shape[1]) * ((X - np.dot(D, A))**2).sum()
    return cost     

def wrapper_score_proposals(this_df, D, feat_stack, params, feat_size=16):
    """ Wrappper for score_proposals routine.
    """
    sidx = np.arange(this_df['f-init'], 
                     this_df['f-init'] + this_df['n-frames'], feat_size)
    X = feat_stack[sidx, :]
    X = normalize(X, axis=1, norm='l2')
    this_score = score_proposals(X, D, params)
    this_df['score'] = this_score
    return this_df
