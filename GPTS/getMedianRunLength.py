import numpy as np


def getMedianRunLength(S):

    T = S.shape[1]

    cdf = np.cumsum(S, axis=0)

    Mrun = np.zeros(T)
    for ii in range(T):
        Mrun[ii] = np.where(cdf[:, ii] >= 0.5)[0][0]

    MchangeTime = np.asarray(range(T)) - Mrun + 1

    return Mrun, MchangeTime
