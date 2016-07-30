import argparse
import os

import cPickle as pkl

import h5py
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


from sparseprop.feature import C3D as FeatHelper
from sparseprop.utils import get_typical_durations
from sparseprop.train import learn_class_independent_model
from sparseprop.train import learn_class_induced_model


def load_dataset(df, hdf5_filename, n_clusters=256, 
                 output_filename=None, verbose=True):
    """Load dataset containing trimmed instances.
    
    Parameters
    ----------
    df : DataFrame
        Dataframe containing the annotations info. It must 
        contain the following fields: 'video-name', 'f-init', 'n-frames'
    hdf5_filename : str
        String containing the path to the HDF5 file containing 
        the features for each video. The HDF5 file must contain 
        a group for each video where the id of the group is the name 
        of the video; and each group must contain a dataset containing
        the features.
    n_clusters : int, optional
        Number of cluster for KMeans
    output_filename : str, optional
        String containing the path to a pickle file where the dataset 
        will be stored. If the file exists, the function will skip 
        the re-compute of the dataset.
    verbose : bool, optional
        Activates verbosity.
    
    Outputs
    -------
    dataset : dict
        Dictionary containing the packed dataset containing the following 
        keys: 'feat' [ndarray containing the feature matrix]
              'label' [ndarray containing the label matrix]
              'video-name' [1darray containing the video name]
              'centers' [ndarray containing the KMeans centers]
    """
    # Avoid re-computing if dataset exists.
    if output_filename:
        if os.path.exists(output_filename):
            with open(output_filename, 'rb') as fobj:
                return pkl.load(fobj)
    
    # Iterate over each annotation instance and load its features.
    video_lst, label_lst, feat_lst = [], [], []
    feat_obj = FeatHelper(hdf5_filename)
    feat_obj.open_instance()
    for k, row in df.iterrows():
        try:
            this_feat = feat_obj.read_feat(row['video-name'],
                                           int(row['f-init']),
                                           int(row['n-frames']))
            feat_lst.append(this_feat)
            label_lst.append(np.repeat(row['label-idx'], this_feat.shape[0]))
            video_lst.append(np.repeat(row['video-name'], this_feat.shape[0]))
        except:
            if verbose:
                print ('Warning: instance from video '
                       '{} was discarded.').format(row['video-name'])
    
    # Stack features in a matrix.
    feat_stack = np.vstack(feat_lst)
    
    # Compute KMeans centers.
    km = KMeans(n_clusters=n_clusters, n_jobs=-1)
    n_samples = np.minimum(1e4, feat_stack.shape[0])
    sidx = np.random.permutation(np.arange(feat_stack.shape[0]))[:n_samples]
    km.fit(feat_stack[sidx, :])
    
    # Pack dataset in a dictionary.
    dataset = {'feat': feat_stack,
               'label': np.hstack(label_lst),
               'video-name': np.hstack(video_lst),
               'centers': km.cluster_centers_}
    
    # Save if desired.
    if output_filename:
        with open(output_filename, 'wb') as fobj:
            pkl.dump(dataset, fobj)
            
    return dataset


def input_parsing():
    """Returns parsed script arguments."""
    description = 'Train an Action SparseProposal model.'
    p = argparse.ArgumentParser(description=description)

    # Specifying training file format.
    p.add_argument('train_filename', type=str,
                   help=('CSV file containing a list of videos and '
                         'temporal annotations. The file must have the '
                         'following headers and format: \n'
                         'video-name f-init n-frames '
                         'video-frames label-idx'))

    # Specifying feature file format.
    p.add_argument('feature_filename', type=str,
                   help=('HDF5 file containing the features for each video '
                         'in `train_filename`. The HDf5 must be formmated as '
                         'follows: (1) Each video is encoded in a group '
                         'where its ID is `video-name`; (2) Inside each '
                         'group there should be an HDF5 dataset containing '
                         'a 2d numpy array with the features.'))

    # Where the model will be saved.
    p.add_argument('model_filename', type=str,
                   help='Pickle file that will contain the learned model.')

    # Optional arguments.
    p.add_argument('--dict_type', type=str, default='induced',
                   help='Type of dictionary: induced, independent')
    p.add_argument('--dict_size', type=int, default=256,
                   help='Size of the sparse dictionary D.')
    p.add_argument('--dataset_filename', type=str, 
                   default=os.path.join('data', 'train_dataset.pkl'),
                   help='Pickle file where the dataset will be stored.')
    p.add_argument('--verbose', action='store_true',
                   help='Activates verbosity.')
    args = p.parse_args()
    return args

def main(train_filename, feature_filename, model_filename, dict_size=256, 
         dict_type='induced', dataset_filename=None, verbose=True):
    """Main subroutine that controls the training procedure and save the 
    trained model to disk. See `input_parsing` for info about the inputs.

    Outputs
    -------
    model : dict
        Dictionary containing the learned model.
        Keys: 
            'D': 2darray containing the sparse dictionary.
            'cost': Cost function at the last iteration.
            'durations': 1darray containing typical durations (n-frames)
                 in the training set.
            'type': Dictionary type.
    """

    ###########################################################################
    # Prepare input/output files.
    ###########################################################################
    # Reading training file.
    if not os.path.exists(train_filename):
        raise RuntimeError('Please provide a valid train file: not exists')
    train_df = pd.read_csv(train_filename, sep=' ')
    rfields = ['video-name', 'f-init', 'n-frames', 'video-frames', 'label-idx']
    efields = np.unique(train_df.columns)
    if not all([field in efields for field in rfields]):
        raise RuntimeError('Please provide a valid train file: bad formatting')
    # Feature file sanity check.
    with h5py.File(feature_filename) as fobj:
        # Checks that feature file contains all the videos in train_filename.
        evideos = fobj.keys()
        rvideos = np.unique(train_df['video-name'].values)
        if not all([x in evideos for x in rvideos]):
            raise RuntimeError(('Please provide a valid feature file: '
                                'some videos are missing.'))

    ###########################################################################
    # Preprocessing.
    ###########################################################################
    if verbose:
        print '[Preprocessing] Starting to preprocess the dataset...'
    # Remove ambiguous segments in train dataframe.
    train_df = train_df[train_df['label-idx']!=-1].reset_index(drop=True)
    # Get dataset.
    dataset = load_dataset(train_df, feature_filename, n_clusters=dict_size, 
                           output_filename=dataset_filename, verbose=verbose)
    dataset['durations'] = get_typical_durations(train_df['n-frames'])
    # Normalize KMeans centers.
    dataset['centers'] = normalize(dataset['centers'], axis=1, norm='l2')
    dataset['feat'] = normalize(dataset['feat'], axis=1, norm='l2')
    # Unifying matrix definitions.
    X, D_0 = dataset['feat'], dataset['centers']
    Y = LabelBinarizer().fit_transform(dataset['label'])
    if verbose:
        print '[Preprocessing] Dataset sucessfully loaded and pre-proccessed.'
    
    ###########################################################################
    # Train
    ###########################################################################
    if verbose:
        print '[Model] Starting to learn the model...'
    if dict_type.lower() == 'independent':
        D, A, cost = learn_class_independent_model(X, D_0, verbose=verbose)
    elif dict_type.lower() == 'induced':
        D, A, W, cost = learn_class_induced_model(X, D_0, Y, verbose=verbose)
    else:
        raise RuntimeError('Please provide a valid type of dictionary.')
    if not os.path.exists(os.path.dirname(model_filename)):
        os.makedirs(os.path.dirname(model_filename))

    # Pack and save model.
    model = {'D': D, 'cost': cost, 
             'durations': dataset['durations'], 'type': dict_type.lower()}
    if dict_type.lower() == 'induced': model['W'] = W
    with open(model_filename, 'wb') as fobj:
        pkl.dump(model, fobj)
    if verbose:
        print '[Model] Model sucessfully saved at {}'.format(model_filename)

if __name__ == '__main__':
    args = input_parsing()
    main(**vars(args))
