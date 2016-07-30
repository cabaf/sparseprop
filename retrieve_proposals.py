import argparse
import os

import cPickle as pkl

import h5py
import numpy as np
import pandas as pd

from sparseprop.retrieve import retrieve_proposals
from sparseprop.utils import wrapper_nms

def input_parsing():
    """Returns parsed script arguments."""
    description = 'Retrieve action proposals using an SparseProposal model'
    p = argparse.ArgumentParser(description=description)

    # Specifying filename_lst format.
    p.add_argument('filename_lst', type=str,
                   help=('CSV file containing a list of videos with '
                         'the number of frames. The file must have the '
                         'following headers and format: \n'
                         'video-name video-frames'))

    # Specifying feature file format.
    p.add_argument('feature_filename', type=str,
                   help=('HDF5 file containing the features for each video '
                         'in `filename_lst`. The HDf5 must be formmated as '
                         'follows: (1) Each video is encoded in a group '
                         'where its ID is `video-name`; (2) Inside each '
                         'group there should be an HDF5 dataset containing '
                         'a 2d numpy array with the features.'))

    # Specifying model format.
    p.add_argument('model_filename', type=str,
                   help=('cPickle file containing an SparseProposal model. '
                         'See `run_train.py` for further details '
                         'about the model format.'))

    p.add_argument('proposal_filename', type=str,
                   help=('CSV file containing the resulting proposals.'))

    p.add_argument('--nms', default=0.65, type=float,
                   help=('Non-maxima supression threshold.'))

    args = p.parse_args()
    return args

def main(filename_lst, feature_filename, model_filename, 
         proposal_filename, nms=0.65, verbose=True):
    """Main subroutine that controls the proposal extraction procedure 
    and save the proposals to disk. See `input_parsing` for info 
    about the inputs."""

    ###########################################################################
    # Prepare input/output files.
    ###########################################################################
    # Reading training file.
    if not os.path.exists(filename_lst):
        raise RuntimeError('Please provide a valid file: not exists')
    df = pd.read_csv(filename_lst, sep=' ')
    rfields = ['video-name', 'video-frames']
    efields = np.unique(df.columns)
    if not all([field in efields for field in rfields]):
        raise RuntimeError('Please provide a valid file: bad formatting')
    # Feature file sanity check.
    with h5py.File(feature_filename) as fobj:
        # Checks that feature file contains all the videos in train_filename.
        evideos = fobj.keys()
        rvideos = np.unique(df['video-name'].values)
        if not all([x in evideos for x in rvideos]):
            raise RuntimeError(('Please provide a valid feature file: '
                                'some videos are missing.'))
    with open(model_filename, 'rb') as fobj:
        model = pkl.load(fobj)
    
    ###########################################################################
    # Retrieve proposals.
    ###########################################################################
    proposal_lst = []
    for k, video_info in df.iterrows():
        prop = retrieve_proposals(video_info, model, feature_filename)
        if nms:
            prop = wrapper_nms(prop)
        proposal_lst.append(prop)
        if verbose:
            print ('Retrieving log: \n\tVideo name: {}'
                   '\n\tNo. Proposals: {}\n\t'.format(video_info['video-name'],
                                                      prop.shape[0]))
    proposal_df = pd.concat(proposal_lst, axis=0)
    # Save results.
    proposal_df.to_csv(proposal_filename, sep=' ', index=False)
    if verbose:
        'Proposals successfully saved at: {}'.format(proposal_filename)

if __name__ == '__main__':
    args = input_parsing()
    main(**vars(args))
