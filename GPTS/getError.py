#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def normlogpdf(x, mu=None, sigma=None):
    if mu is None:
        mu = 0
    if sigma is None:
        sigma = 1

  # Return nan for out of range parameters.

    if isinstance(sigma, np.ndarray):
        sigma[sigma <= 0] = np.nan

    try:
        y = -0.5 * ((x - mu) / sigma) ** 2 - np.log(np.sqrt(2 * np.pi)
                                                    * sigma)
    except:
        raise Error('stats:normpdf:InputSizeMismatch: Non-scalar arguments must match in size.'
                    )

    return y


def getError(
    observed,
    predicted,
    mu=None,
    s2=None,
):
    MSE = np.mean((observed - predicted) ** 2)
    MAE = np.mean(np.abs(observed - predicted))
    NLL = None

    if mu and s2:
        MSE = np.mean((observed - mu * np.ones_like(observed)) ** 2)
        MAE = np.mean(np.abs(observed - mu * np.ones_like(observed)))
        NLL = -normlogpdf(observed, mu, np.sqrt(s2))

    return (NLL, MSE, MAE)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    Y = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    P = np.array([-1.1, -0.2, 0.3, 0.4, 1.6])

    (NLL, MSE, MAE) = getError(Y, P)
    if NLL is None and np.allclose(MSE, 0.112) and np.allclose(MAE,
                                                               0.28):
        print 'getError Test 1 PASSED'
    else:
        print 'getError Test 1 FAILED'
    (NLL, MSE, MAE) = getError(Y, P, np.mean(P), np.var(P))
    if np.allclose(np.asarray([1.72219566, 1.10691068, 0.8154599,
                   0.84784332, 1.20406094]), NLL) and np.allclose(0.54,
                                                                  MSE) and np.allclose(0.64, MAE):
        print 'getError Test 2 PASSED'
    else:
        print 'getError Test 2 FAILED'
