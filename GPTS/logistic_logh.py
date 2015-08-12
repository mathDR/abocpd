#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from logistic import *


def logistic_logh(v, theta_h):
    # h(t) = h * logistic(at + b)
    # theta_h: [logit(h), a, b]
    # derived on p. 230 - 232 of DPI notebook

    if np.isscalar(v):
        v = np.asarray([v])
    T = v.shape[0]
    if len(v.shape) == 2:
        v = np.reshape(v, (T, ))

    (h, a, b) = (logistic(theta_h[0]), theta_h[1], theta_h[2])

    logmh = loglogistic(-theta_h[0])
    logh = loglogistic(theta_h[0])

    logisticNeg = logistic(-a * v - b)

    logH = loglogistic(a * v + b) + logh

    logmH = logsumexp(-a * v - b, logmh) + loglogistic(a * v + b)

    if len(logH.shape) == 1:
        logH = np.atleast_2d(logH).T

    if len(logmH.shape) == 1:
        logmH = np.atleast_2d(logmH).T

    # Derivatives

    dlogH = np.zeros((T, 3))
    dlogH[:, 0] = logistic(-theta_h[0])
    dlogH[:, 1] = logisticNeg * v
    dlogH[:, 2] = logisticNeg

    dlogmH = np.zeros((T, 3))
    dlogmH[:, 0] = dlogsumexp(-a * v - b, 0, logmh, -h)
    dlogmH[:, 1] = dlogsumexp(-a * v - b, -v, logmh, 0) + v * logisticNeg
    dlogmH[:, 2] = dlogsumexp(-a * v - b, -1, logmh, 0) + logisticNeg

    assert logH.shape == (T, 1)
    assert logmH.shape == (T, 1)
    assert dlogH.shape == (T, 3)
    assert dlogmH.shape == (T, 3)

    return (logH, logmH, dlogH, dlogmH)


def logsumexp(x, c):

    # function logZ = logsumexp(x, c)

    maxx = np.max(x, c)
    if isinstance(maxx, np.ndarray):
        maxx[np.isnan(maxx)] = 0
        maxx[np.logical_not(np.isfinite(maxx))] = 0
    return np.log(np.exp(x - maxx) + np.exp(c - maxx)) + maxx


def dlogsumexp(
    x,
    dx,
    c,
    dc,
):

    maxx = np.max([np.max(x), np.max(c)])
    if isinstance(maxx, np.ndarray):
        maxx[np.isnan(maxx)] = 0
        maxx[np.logical_not(np.isfinite(maxx))] = 0

    # TODO rewrite to avoid 0/0 in extreme cases

    return (np.exp(x - maxx) * dx + np.exp(c - maxx) * dc) / (np.exp(x
                                                                     - maxx) + np.exp(c - maxx))


def loglogistic(x):

    # function y = loglogistic(x)

    if isinstance(x, float):
        if x < 0:
            y = -np.log(np.exp(x) + 1.0) + x
        else:
            y = -np.log(1.0 + np.exp(-x))
    else:
        y = np.zeros_like(x)
        negx = x < 0
        nnegx = x >= 0
        y[negx] = -np.log(np.exp(x[negx]) + 1.0) + x[negx]
        y[nnegx] = -np.log(1.0 + np.exp(-x[nnegx]))
    return y


if __name__ == '__main__':
    from numpy import asarray, allclose
    v = asarray(range(1, 4))
    theta_h = asarray([1.5, 2.5, .5])
    (logH, logmH, dlogH, dlogmH) = logistic_logh(v, theta_h)

    assert allclose(logH, asarray([[-0.25000063], [-0.20549172],
                    [-0.20174868]]))
    assert allclose(logmH, asarray([[-1.50868933], [-1.68333656],
                    [-1.69991147]]))
    assert allclose(dlogH, asarray([[0.18242552, 0.04742587,
                    0.04742587], [0.18242552, 0.00814028, 0.00407014],
                    [0.18242552, 0.00100605, 0.00033535]]))
    assert allclose(dlogmH, asarray([[-0.64228408, -0.16697709,
                    -0.16697709], [-0.79966016, -0.0356828,
                    -0.0178414], [-0.8160738, -0.00450053,
                    -0.00150018]]))
    print 'logistic_logh Test PASSED'
