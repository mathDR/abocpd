#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from copy import copy


def stdSplit(Z, X):
    if isinstance(X, int):
        Ttrain = X
    else:
        Ttrain = X.shape[0]
    Y = copy(Z)

    if len(Y.shape) == 1:
        Y = np.atleast_2d(Y).T

    Ytrain = Y[:Ttrain, :]
    Ytest = Y[Ttrain:, :]

    col_means = np.mean(Ytrain, axis=0)
    # To match matlab std:
    col_stds = np.sqrt(
        ((Ytrain - col_means) ** 2).sum(axis=0) / (Ytrain.shape[0] - 1))
    #col_stds = np.std(Ytrain, axis=0)

    Ytrain -= col_means[:, np.newaxis]
    Ytrain /= col_stds[:, np.newaxis]

    Ytest -= col_means[:, np.newaxis]
    Ytest /= col_stds[:, np.newaxis]

    return (Ytrain, Ytest)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    Y = 10.0 * np.random.randn(1000, 1)
    N = 100
    mu = np.mean(Y[:N, :])
    sig = np.std(Y[:N, :])
    v = (Y[:N, :] - mu) / sig

    (Ytrain, Ytest) = stdSplit(Y, N)
    assert np.allclose(Ytrain, v)
    v = (Y[N:, :] - mu) / sig
    assert np.allclose(Ytest, v)
    print 'stdSplit Test PASSED'
