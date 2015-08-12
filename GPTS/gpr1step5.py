#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from dreverseYuleWalkerES import dreverseYuleWalkerES
import pyGPs

def gpr1step5(
    logtheta,
    model,
    T,
    dt,
    ):

# function [alpha, sigma, dalpha, dsigma] = gpr1step5(logtheta, covfunc, T, dt)

  # if ischar(covfunc), covfunc = cellstr(covfunc) end # convert to cell if needed

    epsilon = 1e-8  # cutoff for change in alpha. []
    minLen = 20  # Must be at least 2, otherwise indexing errors will result.

    D = 1
    if len(model.covfunc.hyp) != logtheta.shape[0]:
        raise Error('Error: Number of parameters do not agree with covfunc in gpr1step5')

    model.covfunc.hyp = logtheta
    Kss = model.covfunc.evaluate(np.atleast_2d(range(1, T)).T / dt, 'diag')[0, 0]

    dKss = np.zeros((1, logtheta.shape[0]))
    x = (np.atleast_2d(range(1, T)) / dt).T
    z = np.zeros((1, 1))
    Kstar = model.covfunc.evaluate(x,z)
    dKstar = np.zeros((T - 1, logtheta.shape[0]))

    for ii in range(logtheta.shape[0]):
        dK = model.covfunc.evaluate((np.atleast_2d(range(T)).T / dt) , der=ii)
        dKss[0, ii] = dK[0, 0]
        dKstar[:, ii] = dK[1:, 0]


    (alpha, dalpha) = dreverseYuleWalkerES(
        Kss,
        Kstar,
        dKss,
        dKstar,
        minLen,
        epsilon,
        )

    pruneLen = alpha.shape[0]
    sigma  = Kss - np.dot(alpha, Kstar[:pruneLen, 0])
    dsigma = np.zeros((pruneLen + 1, len(logtheta)))
    for ii in range(logtheta.shape[0]):
        dsigma[1:, ii] = dKss[0, ii] - (np.dot(dalpha[:, :, ii], Kstar[:pruneLen, 0]) + \
            np.dot(alpha, dKstar[:pruneLen, ii]))

  # Add in the prior preditictive in the first row, are these memory ineffiecient
  # operations??

    alpha  = np.concatenate((np.zeros((1, pruneLen)), alpha))
    dalpha = np.concatenate((np.zeros((1, pruneLen, len(logtheta))),dalpha))
    sigma = np.append(Kss, sigma)
    dsigma[0, :] = dKss[0, :]

  # TODO note that these should be >= noise variance

    assert (sigma >= 0).all()
    return (alpha, np.atleast_2d(sigma).T, dalpha, dsigma)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    N = 1000
    dt = 2 * np.pi / N
    Ttrain = np.atleast_2d(range(int(.2 * N))).T * dt
    Xtrain = np.sin(Ttrain)# + 0.1 * np.random.normal(0, 1, Ttrain.shape)
    Ttest = np.atleast_2d(range(int(.2 * N), N)).T * dt
    Xtest = np.sin(Ttest)# + 0.1 * np.random.normal(0, 1, Ttest.shape)

    covfunc = pyGPs.cov.RQ() + pyGPs.cov.Const() + pyGPs.cov.Noise()
    #covfunc = pyGPs.cov.RBF()
    model = pyGPs.GPR()
    model.setPrior(kernel=covfunc)
    model.setScalePrior([1.0, 1.0])

    #model.optimize(Ttrain,Xtrain)
    #model.covfunc.hyp = np.asarray([0.2803317208640337, -0.1696694454864699])
    model.covfunc.hyp = np.asarray([0.6079970164514539,
                                   0.1891319698857131,
                                   0.20521371431434787,
                                   0.016186641234573418,
                                   -2.2595732692553603])

    logtheta = model.covfunc.hyp
    dt = 1
    alpha, sigma, dalpha, dsigma = gpr1step5(logtheta,model,500,dt)
