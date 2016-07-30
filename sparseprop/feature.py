import h5py
import numpy as np

"""
Once you extract C3D codes of your videos, you should save it as HDF5. We save a Group for each video and one Dataset with C3D features on each group. The name of each Group corresponds to the video-id, while the name used for Dataset is c3d-features. Do not forget to set up accordingly your stride and pooling strategy.
"""

class C3D(object):
    def __init__(self, filename, feat_id='c3d_features',
                 t_size=16, t_stride=16, pool_type=None):
        """
        Parameters
        ----------
        filename : str.
            Full path to the hdf5 file.
        feat_id : str, optional.
            Dataset identifier.
        t_size : int, optional.
            Size of temporal receptive field C3D-model.
        t_stride : int, optional.
            Size of temporal stride between features.
        pool_type : str, optional.
            Global pooling strategy over a bunch of features.
            'mean', 'max'
        """
        self.filename = filename
        with h5py.File(self.filename, 'r') as fobj:
            if not fobj:
                raise ValueError('Invalid type of file.')
        self.feat_id = feat_id
        self.fobj = None
        self.t_size = t_size
        self.t_stride = t_stride
        self.pool_type = pool_type

    def open_instance(self):
        """Open file and keep it open till a close call.
        """
        self.fobj = h5py.File(self.filename, 'r')

    def close_instance(self):
        """Close existing h5py object instance.
        """
        if not self.fobj:
            raise ValueError('The object instance is not open.')
        self.fobj.close()
        self.fobj = None

    def read_feat(self, video_name, f_init=None, duration=None,
                  return_reshaped=True):
        """Read C3D features and stack them into memory.
        Parameters
        ----------
        video-name : str.
            Video identifier.
        f_init : int, optional.
            Initial frame index. By default the feature is
            sliced from frame 1.
        duration : int, optional.
            Duration in term of number of frames. By default
            it is set till the last feature.
        return_reshaped : bool.
            Return stack of features reshaped when pooling is applied.
        """
        if not self.fobj:
            raise ValueError('The object instance is not open.')
        s = self.t_stride
        t_size = self.t_size
        if f_init and duration:
            frames_of_interest = range(f_init, 
                                       f_init + duration - t_size + 1, s)
            feat = self.fobj[video_name][self.feat_id][frames_of_interest, :]
        elif f_init and (not duration):
            feat = self.fobj[video_name][self.feat_id][f_init:-t_size+1:s, :]
        elif (not f_init) and duration:
            feat = self.fobj[video_name][self.feat_id][:duration-t_size+1:s, :]
        else:
            feat = self.fobj[video_name][self.feat_id][:-t_size+1:s, :]
        pooled_feat = self._feature_pooling(feat)

        if not return_reshaped:
            feat_dim = feat.shape[1]
            pooled_feat = pooled_feat.reshape((-1, feat_dim))
            if not pooled_feat.flags['C_CONTIGUOUS']:
                return np.ascontigousarray(pooled_feat)
        return pooled_feat

    def _feature_pooling(self, x):
        """Compute pooling of a feature vector.
        Parameters
        ----------
        x : ndarray.
            [m x d] array of features.m is the number of features and
            d is the dimensionality of the feature space.
        """
        if x.ndim != 2:
            raise ValueError('Invalid input ndarray. Input must be [mxd].')

        if not self.pool_type:
            return x

        if self.pool_type == 'mean':
            return x.mean(axis=0)
        elif self.pool_type == 'max':
            return x.max(axis=0)
