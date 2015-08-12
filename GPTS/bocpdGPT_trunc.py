#!/usr/bin/python
# -*- coding: utf-8 -*-
# Ryan Turner(rt324@cam.ac.uk)
# Yunus Saatci(ys267@cam.ac.uk)
#
# Inputs:
# X  T x 1  variable containing the time series.
# covfunc  gpr compatible covariance function
# theta_m  param_count x 1  covfunc loghypers
# theta_h  hazard_param_count x 1  parameters to logistic_h2() hazard function
# scalePrior  2 x 1  log shape and log inverse scale for gamma prior on cov prec
# dt  1 x 1  time between data points as seen by the GP covariance function
#
# Outputs:
# R  T + 1 x T + 1  Object giving the run - length probabilities, R(r, t) =
#   P(runlength_t - 1=r - 1 | X_1: t - 1). [P]
# S  T x T  Object giving the run - length probabilities, R(r, t) =
#   P(runlength_t - 1=r - 1 | X_1: t - 1). [P]
# nlml  1 x 1  negative log marginal likelihood of the data, X(1: end), under the
#   model = P(X_1: T), integrating out all the runlengths. [log P]
# Z  T x 1  1 step ahead predictive distribution: Z(t) = p(X(t) | X(1: t - 1)). [P / X]
#
# Will be equivalent to:
# bocpdGPT(X, covfunc, true_model_params + log([dt 1 1]'), true_hazard_params,
# [true_scale_params 0], 1)
# if +dt is applied to every time unit hyper parameter.

import numpy as np
from Utils.iskosher import isKosher
from Utils.gpr1step5 import gpr1step5
from Utils.MoTrnd import MoTrnd
from Hazards.logistic_h2 import logistic_h2
from studentpdf import studentpdf


def bocpdGPT_trunc(
    X,
    model,
    theta_m,
    theta_h,
    scalePrior,
    dt,
):

    # Maximum numbers of points considered for predicting the next one regardless of
    # the run length and cov function. Set to Inf is we don't care about speed.

    maxPossibleLen = 500

    num_hazard_params = len(theta_h)
    num_model_params = len(theta_m)

    assert isKosher(X)
    assert dt > 0

    (T, D) = X.shape

    # Number of time point observed. 1 x 1. [s]
    # TODO extend to higher D

    assert D == 1

    # Never need to consider more than T points in the past. 1 x 1. [points]

    maxPossibleLen = min(T, maxPossibleLen)

    # Ensure the gamma prior parameters are positive(as required). 2 x 1. []

    scalePrior = np.exp(scalePrior)
    alpha0 = scalePrior[0]
    beta0 = scalePrior[1]

    # Precompute all the gpr aspects of algorithm. [maxLen x maxLen, maxLen x
    # 1]

    (alpha, sigma2, dalpha, dsigma2) = gpr1step5(theta_m, model,
                                                 maxPossibleLen, dt)

    maxLen = alpha.shape[0]
    assert maxLen >= 1

    # Evaluate the hazard function:

    # H(r) = P(runlength_t=0 | runlength_t - 1=r - 1)
    # Pre - computed the hazard in preperation for steps 4 & 5, alg 1, of[RPA]

    (H, dH) = logistic_h2(np.asarray(range(1, maxLen + 1)), theta_h)

    R = np.zeros((maxLen + 1, T + 1))

    # The standardized square error for each runlength.
    SSE = np.zeros((maxLen, D))

    # The evidence at each time step = > Z(t) = P(X_t | X_1: t - 1).
    Z = np.zeros((T, 1))
    predMeans = np.zeros((T, 1))
    predMed = np.zeros((T, 1))

    # At time t = 1, we have complete knowledge about the run length. This assumes
    # there was surely a change point right before the first data point not at the
    # first data point. Implements step 1, alg 1, of[RPA].
    # = > P(runglenth_0=0 | nothing) = 1

    R[0, 0] = 1

    # Initialize first SSE to contribution from gamma prior.

    SSE[0] = 2 * beta0

    # How many degrees of freedom in the prediction for each run length.

    df = np.asarray([2 * alpha0]) + np.asarray(range(maxLen))

    for t in range(1, T + 1):
    # Implictly Implements step 2, alg 1, of[RPA]: oberserve new datum, simply
    # by incrementing the loop index.

    # Evaluate the predictive distribution for the new datum under each of the
    # parameters. Implements step 3, alg 1, of[RPA]. predprobs(r)
    # = p(X(t) | X(1: t - 1), runlength_t - 1=r - 1). t x 1. [P]

        predprobs = np.zeros(maxLen)
        if t < maxLen:
            mu = np.dot(alpha[:t, :t], X[:t, 0][::-1])

            # The predictive variance for each prediction
            pred_var = sigma2[:t, 0] * SSE[:t, 0] / df[:t]

            # get the posterior predictive probability for each run length
            predprobs[:t] = studentpdf(X[t - 1, 0], mu, pred_var, df[:t], 1)

            # Update the SSE for each run length
            SSE[1:t + 1, 0] = SSE[:t, 0] + \
                (mu - X[t - 1, 0]) ** 2 / sigma2[:t, 0]
            SSE[0, 0] = 2 * beta0  # 1 x 1. []
        else:
            mu = np.dot(alpha, X[t - maxLen + 1:t, 0][::-1])

            # The predictive variance for each prediction
            pred_var = sigma2[:, 0] * SSE[:, 0] / df

            # get the posterior predictive probability for each run length
            predprobs = studentpdf(X[t - 1, 0], mu, pred_var, df, 1)

            # Update the SSE for each run length
            SSE[1:maxLen, 0] = SSE[:maxLen - 1, 0] + \
                (mu[:maxLen - 1] - X[maxLen - 1, 0]) ** 2 / \
                sigma2[:maxLen - 1, 0]
            SSE[0, 0] = 2 * beta0

        # endif
        predMeans[t - 1] = np.dot(R[:mu.shape[0], t - 1].T, mu)

        predMed[t - 1] = np.median(
            MoTrnd(R[:mu.shape[0], t - 1], mu, pred_var[:mu.shape[0]], df[:mu.shape[0]], 1000))

        # Evaluate the growth probabilities - shift the probabilities up and to the
        # right, scaled by the hazard function and the predictive
        # probabilities.
        R[1:, t] = R[: maxLen, t - 1] * predprobs * (1 - H[: maxLen])

        # Evaluate the probability that there * was * a changepoint and we're
        # accumulating the mass back down at r = 0.

        R[0, t] = (R[: maxLen, t - 1] * predprobs * H[: maxLen]).sum()

        # Renormalize the run length probabilities for improved numerical stability.
        # Note that unlike in [RPA] which keeps track of P(r_t, X_1: t), we keep track
        # of P(r_t | X_1: t) = > unnormalized R(i, t + 1) = P(runlength_t=i - 1 | X_1: t)
        # * P(X_t | X_1: t - 1) = > normalization const Z(t) = P(X_t | X_1: t - 1). Sort of
        # Implements step 6, alg 1, of[RPA].

        Z[t - 1] = R[:, t].sum()

        R[: maxLen, t] /= Z[t - 1]

        R[maxLen - 1, t] = R[maxLen - 1, t] + R[maxLen, t]
        R[maxLen, t] = 0

    # endTloop

    # Get the negative log marginal likelihood of the data, X(1: end), under
    # the model = P(X_1: T), integrating out all the runlengths. 1 x 1. [log
    # P]

    nlml = -sum(np.log(Z))

    return (R, nlml, Z, predMeans, predMed)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pyGPs
    N = 1000
    deltat = 2 * np.pi / N
    Ttrain = np.atleast_2d(range(int(.2 * N))).T * deltat
    Xtrain = np.sin(Ttrain) + 0.1 * np.random.normal(0, 1, Ttrain.shape)
    Ttest = np.atleast_2d(range(int(.2 * N), N + 1)).T * deltat
    Xtest = np.sin(Ttest) + 0.1 * np.random.normal(0, 1, Ttest.shape)

    covfunc = pyGPs.cov.RQ() + pyGPs.cov.Const() + pyGPs.cov.Noise()

    model = pyGPs.GPR()
    model.setPrior(kernel=covfunc)

    # theta_h, theta_m and scalePrior from bocpdGPTlearn

    theta_h = np.asarray([-4.31231611, 0.95020107, 0.9721393])
    theta_m = np.asarray([-0.80060887, 0.23849669, -4.90748963,
                         -0.90833473, -0.6339999])
    scalePrior = np.asarray([3.30619906749, 0.])
    dt = 1

    (R, nlml, Z, predMeans, predMed) = bocpdGPT_trunc(
        Xtest,
        model,
        theta_m,
        theta_h,
        scalePrior,
        dt,
    )
