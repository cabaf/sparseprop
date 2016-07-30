import numpy as np
import spams

""" 
This interface allows the training of Class-Indepedent and Class-Induced 
models - Caba et al. CVPR 2016.
"""


def learn_class_independent_model(X, D, tol=0.01, max_iter=250, 
                                  verbose=True, params=None):
    """Class independent dictionary learning.
    
    Parameters
    ----------
    X : ndarray
        2D numpy array containing an stack of features with shape nxm 
        where n is the number of samples and m the feature dimensionality.
    D : ndarray
        2D numpy array containing an initial guess for the dictionary D.
        Its shape is dxm where d is the number of dictionary elements.
    tol : float, optional
        Global tolerance for optimization convergence.
    max_iter : int, optional
        Maximum number of iterations.
    verbose : bool, optional
        Enable verbosity.
    params : dict, optional
        Dictionary containing the optimization parameters (for Spams).
    """
    if not params:
        params = {'loss': 'square', 'regul': 'l1l2',
                  'numThreads' : -1, 'verbose' : False, 
                  'compute_gram': True, 'ista': True, 'linesearch_mode': 2,
                  'lambda1' : 0.05, 'tol' : 1e-1} 
    X = np.asfortranarray(X.T.copy())
    D = np.asfortranarray(D.T.copy())
    A = np.zeros((D.shape[1], X.shape[1]), order='FORTRAN')
    prev_cost = 1e9
    n_samples = X.shape[1]
    for i in range(1, max_iter + 1):
        # Solves coding step.
        A = spams.fistaFlat(np.sqrt(1.0) * X, 
                            np.sqrt(1.0) * D,
                            A, **params)
        
        # Dictionary update as least square.
        D = np.dot(np.dot(np.linalg.inv(np.dot(A, A.T)), A), X.T).T
        
        # Compute cost.
        cost = (1.0/n_samples) * ((X - np.dot(D, A))**2).sum() + \
            2 * params['lambda1'] * (A**2).sum()
        
        # Check convergence conditions.
        if prev_cost - cost <= tol:
            break
        else:
            prev_cost = cost
        if verbose:
            #if not i % 10:
            print 'Iteration [{}] / Cost function [{}]'.format(i, cost)
    return D.T, A.T, cost


def learn_class_induced_model(X, D, Y, tol=0.01, max_iter=300, 
                              verbose=True, local_params=None, params=None):
    """Class induced dictionary learning.
    
    Parameters
    ----------
    X : ndarray
        2D numpy array containing an stack of features with shape nxm 
        where n is the number of samples and m the feature dimensionality.
    D : ndarray
        2D numpy array containing an initial guess for the dictionary D.
        Its shape is dxm where d is the number of dictionary elements.
    Y : ndarrat
        2D numpy array containing a matrix that maps features and labels.
        Its shape is nxc where c is the number of classes.
    tol : float, optional
        Global tolerance for optimization convergence.
    max_iter : int, optional
        Maximum number of iterations.
    verbose : bool, optional
        Enable verbosity.
    local_params : dict, optional
        Dictionary containing the values of lambda for each optimization term.
    params : dict, optional
        Dictionary containing the optimization parameters (for Spams).
    """
    if not local_params:
        local_params = {'lambda1': 0.05, 'lambda2': 0.05, 'lambda3': 0.025}
    if not params:
        params = {'loss': 'square', 'regul': 'l1l2',
                  'numThreads' : -1, 'verbose' : False,
                  'compute_gram': True, 'ista': True, 'linesearch_mode': 2,
                  'lambda1': local_params['lambda1'], 'tol' : 1e-1} 
    X = np.asfortranarray(X.T.copy())
    D = np.asfortranarray(D.T.copy())
    Y = np.asfortranarray(Y.T.copy())
    n_dict_elem = D.shape[1]
    n_samples = X.shape[1]

    # Initialize A without classification loss.
    A = spams.fistaFlat(X, D,
                        np.zeros((D.shape[1], X.shape[1]), order='FORTRAN'),
                        **params)

    prev_cost = 1e9
    for i in range(1, max_iter + 1):

        # Solves W update.
        rl = local_params['lambda3'] / local_params['lambda2']
        W = np.dot(
                np.linalg.inv(np.dot(A, A.T) + \
                np.diag(np.ones(n_dict_elem) * rl)),
                np.dot(A, Y.T))

        # Solves Dictionary update.
        D = np.dot(np.dot(np.linalg.inv(np.dot(A, A.T)), A), X.T).T

        # Solves coding step.
        U = np.vstack((X, np.sqrt(local_params['lambda2']) * Y))
        V = np.vstack((D, np.sqrt(local_params['lambda2']) * W.T))
        A = spams.fistaFlat(U, V, A, **params)

        # Compute cost.
        cost = (1.0/n_samples) * ((X - np.dot(D, A))**2).sum() + \
            local_params['lambda1'] * (A**2).sum() + \
            local_params['lambda2'] * ((np.dot(W.T, A) - Y)**2).sum() + \
            local_params['lambda3'] * (W**2).sum()

        # Check convergence conditions.
        if prev_cost - cost <= tol:
            break
        else:
            prev_cost = cost
            if verbose:
                #if not i % 10:
                print 'Iteration [{}] / Cost function [{}]'.format(i, cost)
    return D.T, A.T, W.T, cost
