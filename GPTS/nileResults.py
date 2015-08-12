#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import Utils.addSalt as addSalt
import Utils.stdSplit as stdSplit
import Utils.TIMpredict as TIMpredict
import Utils.getError as getError
import Utils.logit as logit

import pyGPs
import time
from GPTSonline import GPTSonline
from bocpdGPTlearn import bocpdGPTlearn
from bocpdGPT import bocpdGPT

from plotS import plotS

if __name__ == '__main__':
    data = \
        np.genfromtxt(
            '/Users/dmarthal/Desktop/source/papers/Thesis/data/nile.txt', delimiter=',')

    idx = [i for i, j in enumerate(data[:, 0]) if j == 715]
    nTtrain = 250
    Y = np.atleast_2d(data[:, 1]).T

  # prec = 1
    prec = 0

    nT = Y.shape[0]
    nTtest = nT - nTtrain

    #Y = addSalt.addSalt(Y, prec)
    # Normalize training data (and test data using mean and std from test data)
    [Ytrain, Ytest] = stdSplit.stdSplit(Y, nTtrain)
    # Concatenate back
    Y = np.concatenate((Ytrain, Ytest))

  # Train TIM
  # set random seed
    randnSeed = 2
    np.random.seed(0)

    muTIM = np.mean(Ytrain, axis=0)
    S2TIM = np.var(Ytrain, axis=0)

  # Test TIM
  # Generate an array of points to compare to Ytest

    predict = np.atleast_2d(TIMpredict.TIMpredict(nTtest, muTIM,
                            S2TIM)).T
    Ttrain = np.atleast_2d(range(nTtrain)).T

  # plt.plot(Ttrain,Ytrain,'r')

    Ttest = np.atleast_2d(range(nTtrain, nT)).T

  # plt.plot(Ttest,Ytest,'b', Ttest,predict, 'g.')
  # plt.show()

  # Baseline error
    #err = getError.getError(Ytest, predict, muTIM, S2TIM)
    # print 'NLL, MSE, MAE:'
    # print err[0], err[1], err[2]

  # Train GPTS
    covFunc = pyGPs.cov.RQ() + pyGPs.cov.Const() + pyGPs.cov.Noise()
    model = pyGPs.GPR()
    model.setPrior(kernel=covFunc)
    #model.setScalePrior([1.0, 1.0])
    # Learn the hyperparameters on the training data
    model.setOptimizer("RTMinimize", 10)
    model.optimize(Ttrain, Ytrain)

  # Do the extrapolation

    logthetaGPTS = model.covfunc.hyp

    #(mu, sig2, df) = GPTSonline(Ytest, covFunc, logthetaGPTS,model.ScalePrior)

  # Plot the stuff
    '''plt.axis([0.0,7.0,-4.0,5.0])
    plt.plot(Ttrain, Ytrain, 'r')
    plt.plot(Ttest, Ytest, 'b-')
    plt.plot(Ttest, mu, 'k-')
    plt.fill_between(Ttest[:, 0], mu[:, 0] + 2. * np.sqrt(sig2[:, 0]),
                     mu[:, 0] - 2. * np.sqrt(sig2[:, 0]), facecolor='g', linewidths=0.0)
    plt.grid()
    plt.show()
    err = getError.getError(Ytest, mu)
    print err[0], err[1]'''

  # Train BOCPD-GPTS
  # np.random.seed(randnSeed)

    theta_h = np.asarray([logit.logit(1.0 / 50.0), 1.0, 1.0])
    dt = 1
    # Learn the change point model
    (theta_h, theta_m, theta_s, nlml) = bocpdGPTlearn(
        Ytrain, model, logthetaGPTS, theta_h, dt)

    model.covfunc.hyp = list(theta_m)
    scalePrior = np.asarray([theta_s, 0.])

    (R, S, nlml, Z, predMeans, predMed) = bocpdGPT(
        Y, model, theta_m, theta_h, scalePrior, dt)

    F = plotS(S, data[:, 1], data[:, 0], idx)
    # plt.subplot(2, 1, 1)
    # plt.plot([nTtrain, nTtrain], [Y.min(), Y.max()], 'g-', linewidth=3)
    # F.ylabel('Water level (mm)')
    # F.subplot(2, 1, 2)
    # F.xlabel('Year')
    plt.show()
    # err = getError(Ytest, predMeans[Ttrain + 1:], predMed[Ttrain + 1:], Z[Ttrain + 1:])
