import numpy as np

def inertia_tensor(X, m = None):
    """
    inertia tensor of a (weighted) point set
    X: array of rank (n,d) where n is the number of points
       and d the dimension
    m: rank (n,) array of masses / weights
    """
    if m is not None:
        X = ((X - np.average(X,axis=0, weights=m)).T/ np.sqrt(m)).T
    else:
        X = X - X.mean(0)

    return dot(transpose(X), X)


