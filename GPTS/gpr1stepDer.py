#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from Utils.dreverseYuleWalkerES import dreverseYuleWalkerES


def gpr1stepDer(
    logtheta,
    covfunc,
    T,
    dt,
    ):

  # function [alpha, sigma, dalpha, dsigma] = gpr1step5(logtheta, covfunc, T, dt)

  # if ischar(covfunc), covfunc = cellstr(covfunc) end # convert to cell if needed

    epsilon = 1e-8  # cutoff for change in alpha. []
    minLen = 20  # Must be at least 2, otherwise indexing errors will result.

    D = 1
    if len(covfunc.hyp) != len(logtheta):
        raise Error('Error: Number of parameters do not agree with covfunc in gpr1step5'
                    )

  # TODO check condition number of cov matrix
  # [Kss, Kstar] = feval(covfunc{:}, logtheta, (1:T - 1)' / dt, 0)

  # X      = np.atleast_2d(range(T-1)).T/dt

    Kss = covfunc.evaluate(np.atleast_2d(range(T)) / dt, 'diag')[0][0]
    x = (np.atleast_2d(range(1, T + 1)) / dt).T
    z = np.zeros((1, 1))
    Kstar = covfunc.evaluate(x, z)  # Dont need this?

    dKss = np.zeros((1, len(logtheta)))
    dKstar = np.zeros((T, len(logtheta)))

    for ii in range(len(logtheta)):
        dK = covfunc.evaluate(x, der=ii)
        dKss[0, ii] = dK[0, 0]
        dKstar[:, ii] = dK[:, ii]

    (alpha, dalpha) = dreverseYuleWalkerES(
        Kss,
        Kstar,
        dKss,
        dKstar,
        minLen,
        epsilon,
        )

    pruneLen = alpha.shape[0]
    sigma = Kss - np.dot(alpha, Kstar[:pruneLen, 0])
    dsigma = np.zeros((pruneLen + 1, len(logtheta)))
    for ii in range(len(logtheta)):
        dsigma[1:, ii] = dKss[ii] - (np.dot(dalpha[:, :, ii], Kstar[:
                pruneLen, 0]) + np.dot(alpha, dKstar[:pruneLen, ii]))

  # Add in the prior preditictive in the first row, are these memory ineffiecient
  # operations??

    alpha = np.concatenate((np.zeros((1, pruneLen)), alpha))
    dalpha = np.concatenate((np.zeros((1, pruneLen, len(logtheta))),
                            dalpha))
    sigma = np.concatenate((Kss, sigma))
    dsigma[0, :] = dKss

  # TODO note that these should be >= noise variance

    assert (sigma > 0).all()
    return (alpha, sigma, dalpha, dsigma)


if __name__ == '__main__':
    import pyGPs
    logtheta = np.log(np.asarray([1., 2.0, 3.0]))

    k = pyGPs.cov.RQ(logtheta[0], logtheta[1], logtheta[2])
    (alpha, sigma2, dalpha, dsigma2) = gpr1stepDer(logtheta, k, 10, 1)
    print alpha, sigma2
