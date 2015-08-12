#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from Utils.reverseYuleWalkerES import reverseYuleWalkerES


def gpr1step(
    logtheta,
    covfunc,
    T,
    dt,
    ):
    epsilon = 1e-8
    minLen = 20  # Must be at least 2, otherwise indexing errors will result.

    D = 1
    if len(covfunc.hyp) != len(logtheta):
        error('Error: Number of parameters do not agree with covariance function in gpr1step'
              )

  # TODO check condition number of cov matrix

    covfunc.hyp = logtheta
    Kss = covfunc.evaluate(np.atleast_2d(range(T)) / dt, 'diag')[0, 0]
    x = (np.atleast_2d(range(1, T + 1)) / dt).T
    z = np.zeros((1, 1))
    Kstar = covfunc.evaluate(x, z)

    alpha = reverseYuleWalkerES(Kss, Kstar, minLen, epsilon)
    pruneLen = alpha.shape[0]
    sigma2 = Kss - np.dot(alpha, Kstar[:pruneLen, 0])

  # Add in the prior preditictive in the first row

    alpha = np.concatenate((np.zeros((1, pruneLen)), alpha), axis=0)
    sigma2 = np.append(Kss, sigma2)
    assert (sigma2 > 0).all()

    return (alpha, np.atleast_2d(sigma2).T)


if __name__ == '__main__':
    import pyGPs
    logtheta = np.log(np.asarray([1., 2.0, 3.0]))

    k = pyGPs.cov.RQ(logtheta[0], logtheta[1], logtheta[2])
    (alpha, sigma2) = gpr1step(logtheta, k, 10, 1)
    print alpha, sigma2
