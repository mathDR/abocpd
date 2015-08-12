#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from gpr1step import gpr1step


def GPTSonline(
    X,
    covfunc,
    loghyper,
    scalePrior=None,
    ):

  # function [mu, sigma2, df] = GPTSonline(X, covfunc, loghyper, scalePrior)
  # Maximum numbers of points considered for predicting the next one regardless of
  # the run length and cov function. Set to Inf is we don't care about speed.

    maxPossibleLen = 50

    assert np.isfinite(X).all() and not np.isnan(X).any()  # Checks that all elements of X are real and finite

    (T, D) = X.shape  # Number of time point observed. 1 x 1. [s]

  # TODO extend to higher D

    assert D == 1

  # Never need to consider more than T points in the past. 1 x 1. [points]

    maxPossibleLen = min(T, maxPossibleLen)

  # set dt = 1 for now

    dt = 1

  # Precompute all the gpr aspects of algorithm. [maxLen x maxLen, maxLen x 1]
  # Should memoize this

    (alpha, sigma2) = gpr1step(loghyper, covfunc, maxPossibleLen, dt)

    maxLen = alpha.shape[0] - 1

  # Extend sigma2 to account for that we might call for its value past maxLen

    if sigma2.shape[0] < T:
        sigma2 = np.concatenate((sigma2, sigma2[-1, 0] * np.ones((T
                                - sigma2.shape[0], 1))))

    mu = np.zeros((T, 1))
    df = None

    if scalePrior is None:

    # => certain output scale

        for t in range(1, T):

      # TODO move to a conv operator

            MRC = min(maxLen, t)  # How many points back to look when predicting
            mu[t, 0] = np.dot(np.atleast_2d(alpha[MRC, :MRC - 1]),
                              np.atleast_2d(X[t - 1:t - MRC:-1, 0]).T)  # MRC x 1. [x]
    else:

    # => uncertain output scale

        assert isinstance(scalePrior, list) and len(scalePrior) == 2

        # Ensure the gamma prior parameters are positive (as required). 2 x 1. []
        scalePrior = np.exp(scalePrior)
        
        (alpha0, beta0) = scalePrior
        SSE = 2 * beta0  # 1 x 1. []
        pred_var = np.zeros((T, 1))
        df = np.zeros((T, 1))

        for t in range(1, T):
            MRC = min(maxLen, t)  # How many points back to look when predicting
            mu[t, 0] = np.dot(alpha[MRC, :MRC - 1], np.atleast_2d(X[t
                              - 1:t - MRC:-1, 0]).T)  # MRC x 1. [x]

      # How many degrees of freedom in the prediction for each run length.

            df[t, 0] = 2 * alpha0 + t  # t x 1. [points]

      # The predictive variance for each prediction

            pred_var[t, 0] = sigma2[t - 1, 0] * SSE / df[t - 1, 0]  # t x 1. [x^2]

      # get the posterior predictive probability for each run length
      # Update the SSE for each run length

            SSE += (mu[t - 1, 0] - X[t - 1, 0]) ** 2 / sigma2[t - 1, 0]  # t x 1. []
        sigma2 = pred_var

    return (mu, sigma2, df)


if __name__ == '__main__':
    import pyGPs
    import matplotlib.pyplot as plt
    import time


    # import pylab
    # pylab.ion()

    def get_fig(
        Ttest=None,
        yTest=None,
        mu=None,
        sig2=None,
        ):

        ax.plot(Ttest, yTest, 'r.-')
        ax.plot(Ttest, mu, 'kx-')
        ax.fill_between(Ttest, mu + 2. * np.sqrt(sig2), mu - 2.
                        * np.sqrt(sig2), facecolor='g', linewidths=0.0)
        ax.grid()

        return fig


    N = 1000
    dt = 2 * np.pi / N
    Ttrain = np.atleast_2d(range(int(.2 * N))).T * dt
    Xtrain = np.sin(Ttrain) + 0.1 * np.random.normal(0, 1, Ttrain.shape)
    Ttest = np.atleast_2d(range(int(.2 * N), N)).T * dt
    Xtest = np.sin(Ttest) + 0.1 * np.random.normal(0, 1, Ttest.shape)

    covfunc = pyGPs.cov.RQ() + pyGPs.cov.Const() + pyGPs.cov.Periodic() \
        + pyGPs.cov.Noise()
    model = pyGPs.GPR()
    model.setPrior(kernel=covfunc)

    # model.setScalePrior([1.0,1.0])

    model.optimize(Ttrain, Xtrain)

    logtheta = model.covfunc.hyp

    # logtheta = np.asarray([0.06349670436628252, -0.058022026856730656, 0.4085934193807769, -1.2944100001643977, 0.03946488667187256])

    # Note, optimize above does NOT take into account Scale Prior (yet) so this will be worse...

    (mu, sigma2, df) = GPTSonline(Xtest, covfunc, logtheta,
                                  model.ScalePrior)

    plt.axis([0.0, 7.0, -4.0, 5.0])
    plt.plot(Ttrain, Xtrain)
    plt.plot(Ttest, Xtest, 'r.-')
    plt.plot(Ttest, mu, 'k-')

    plt.fill_between(Ttest[:, 0], mu[:, 0] + 2. * np.sqrt(sigma2[:,
                     0]), mu[:, 0] - 2. * np.sqrt(sigma2[:, 0]),
                     facecolor='g', linewidths=0.0)
    plt.grid()
    plt.show()
