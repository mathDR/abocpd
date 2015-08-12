#!/usr/bin/python
# -*- coding: utf-8 -*-

# Ryan Turner (rt324@cam.ac.uk)
# Yunus Saatci (ys267@cam.ac.uk)
#
# Inputs:
# X  T x 1  variable containing the time series.
# covfunc  gpr compatible covariance function
# theta_m  param_count x 1  covfunc loghypers
# theta_h  hazard_param_count x 1  parameters to logistic_h2() hazard functionimport ipdb; ipdb.set_trace()import ipdb; ipdb.set_trace()
# theta_s  1 x 1  log shape for gamma prior (with scale = 1) on cov prec
# dt  1 x 1  time between data points as seen by the GP covariance function
#
# Outputs:
# Learned hypers:
# hazard_params  param_count x 1  covfunc loghypers
# model_params  hazard_param_count x 1  parameters to logistic_h2() hazard
# scale_params  1 x 1  log shape for gamma prior (with scale = 1) on cov prec
# nlml  1 x 1  negative log marginal likelihood of the data, X(1:end), under the
#   model = P(X_1:T), integrating out all the runlengths. [log P]

import numpy as np
import pyGPs
from pyGPs.Optimization.rt_minimize import rt_minimize

from Utils.rmult import rmult
from Utils.logsumexp import logsumexp
from Utils.iskosher import isKosher

from Hazards.logistic_logh import logistic_logh
from studentlogpdf import studentlogpdf
from Utils.gpr1step5 import gpr1step5


def bocpdGPTlearn(
    X,               # Training data
    model,       # The current GP model
    theta_m,    # the hyperparameters for the GP model
    theta_h,     # the hyperparameters for the hazard function
    dt=1,         # the timestep
):

    max_minimize_iter = 30
    num_hazard_params = len(theta_h)
    if model.ScalePrior:
        theta_s = model.ScalePrior[
            0]  # alpha from the prior on scale (assumed beta is identity)
    else:
        theta_s = 0

    theta = np.append(np.append(theta_h, theta_m), theta_s)

    (theta, nlml, i) = rt_minimize(
        theta,
        dbocpdGP,
        -max_minimize_iter,
        X,
        model,
        num_hazard_params,
        dt,
    )

    hazard_params = theta[:num_hazard_params]
    model_params = theta[num_hazard_params:-1]
    scale_params = theta[-1]

    return (hazard_params, model_params, scale_params, nlml[-1])


def dbocpdGP(
    theta,
    X,
    model,
    num_hazard_params,
    dt,
):

    beta0 = 1
    num_scale_params = 1

  # Maximum numbers of points considered for predicting the next one regardless of
  # the run length and cov function. Set to Inf is we don't care about speed.

    maxPossibleLen = 500

    theta_h = theta[:num_hazard_params]  # num_hazard x 1
    theta_m = theta[num_hazard_params:-1]  # num_model x 1
    alpha0 = np.exp(theta[-1])  # Use exp to ensure it is positive. 1 x 1
    num_model_params = len(theta_m)  # 1 x 1

    assert dt > 0

    (T, D) = X.shape  # Number of time point observed

    assert D == 1

  # Never need to consider more than T points in the past.

    maxPossibleLen = min(T, maxPossibleLen)

  # Evaluate the hazard function for this interval.
  # H(r) = P(runlength_t = 0|runlength_t-1 = r-1)
  # Pre-computed the hazard in preperation for steps 4 & 5, alg 1, of [RPA]
  # logH = log(H), logmH = log(1-H)

    (logH, logmH, dlogH, dlogmH) = logistic_logh(
        np.asarray(range(1, T + 1)), theta_h)
    assert isKosher(dlogH)
    assert isKosher(dlogmH)

  # R(r, t) = P(runlength_t-1 = r-1|X_1:t-1).
  # P(runglenth_0 = 0|nothing) = 1 => logR(1, 1) = 0

    logR = np.zeros((T + 1, 1))

    # pre-allocate the run length distribution. [P]
    dlogR_h = np.zeros((T + 1, num_hazard_params))
    dlogR_m = np.zeros((T + 1, num_model_params))
    dlogR_s = np.zeros((T + 1, num_scale_params))

    SSE = np.zeros((T + 1, D))

    # This will change with higher D
    dSSE = np.zeros((T + 1, num_model_params))

    SSE[0, 0] = 2 * beta0  # 1 x 1

  # Pre-compute GP stuff:

    (alpha, sigma2, dalpha, dsigma2) = gpr1step5(theta_m, model,
                                                 maxPossibleLen, dt)
    maxLen = alpha.shape[0]

    # Extend sigma2 to account for that we might call for its value past maxLen
    # t - maxLen x 1

    sigma2 = np.concatenate((sigma2, sigma2[-1, 0] * np.ones((T
                            - sigma2.shape[0], 1))))

    dsigma2 = np.concatenate((dsigma2, np.tile(dsigma2[-1, :], (T
                             - maxLen, 1))))

    ddf = 2

    for t in range(1, T + 1):
        MRC = min(maxLen, t)  # How many points back to look when predicting

        mu = np.dot(alpha[:MRC, :MRC - 1], X[
                    t - MRC:t - 1, 0][::-1])  # MRC x 1. [x]

        # Extend the mu (mean) prediction for the older (> MRC) run length
        # hypothesis

        if MRC < t:
            mu = np.append(mu, mu[-1] * np.ones(
                t - mu.shape[0]))  # t - MRC x 1. [x]

        df = np.asarray([2 * alpha0]) + np.asarray(range(t))
        pred_var = sigma2[:t, 0] * SSE[:t, 0] / df
        dpredvar_s = np.atleast_2d(
            ddf * -sigma2[:t, 0] * SSE[:t, 0] / df ** 2).T

        (logpredprobs, dlogpredprobs) = studentlogpdf(X[t - 1, 0], mu,
                                                      pred_var, df, 2)

        # Now do the derivatives. [t x 1, t x 1]

        dmu = np.zeros((t, num_model_params))
        dpredvar = np.zeros((t, num_model_params))

        for ii in range(num_model_params):

        # MRC x 1. [x/theta_m]

            dmu[:MRC, ii] = np.dot(dalpha[:MRC, :MRC - 1, ii], X[t
                                   - MRC:t - 1, 0][::-1])
            if MRC < t:

            # Extend the mu (mean) prediction for the older (>MRC) run length
            # hypothesis

                dmu = np.concatenate((dmu, [dmu[MRC - 1]] * np.ones((t
                                                                     - dmu.shape[0], 1))))

                # Use the product rule. t x 1. [x^2/theta_m]

            dpredvar[:, ii] = (dsigma2[:t, ii] * SSE[:t, 0] + sigma2[:
                               t, 0] * dSSE[:t, ii]) / df

            # Use the quotient rule. t x 1. [1/theta_m]

            dSSE[1:t + 1, ii] = dSSE[:t, ii] + 2 * (mu - X[t - 1, 0]) \
                / sigma2[:t, 0] * dmu[:, ii] + -(mu - X[t - 1, 0]) ** 2 \
                / sigma2[:t, 0] ** 2 * dsigma2[:t, ii]
            dSSE[0, ii] = 0

        dlogpredprobs_m = rmult(dmu, dlogpredprobs[:, 0]) \
            + rmult(dpredvar[:t, :], dlogpredprobs[:, 1])

        # mu has zero dependence on alpha (scale). t x 1. [log(P/x)]

        dlogpredprobs_s = np.atleast_2d(dpredvar_s[:t, 0]
                                        * dlogpredprobs[:, 1] + ddf * dlogpredprobs[:, 2]).T

        # Update with the Maha error of predicting the next point. t x 1. []

        SSE[1:t + 1, 0] = SSE[:t, 0] + (mu - X[t - 1, 0]) ** 2 \
            / sigma2[:t, 0]
        SSE[0, 0] = 2 * beta0  # 1 x 1. []

        # Update the run length distributions and their derivatives.

        logMsg = logR[:t, 0] + logpredprobs + logH[:t, 0]  # t x 1
        dlogMsg_h = dlogR_h[:t, :] + dlogH[:t, :]  # t x num_hazard

        logR[1:t + 1, 0] = logR[:t, 0] + \
            logpredprobs + logmH[:t, 0]  # t x 1. [P]

        dlogR_h[1:t + 1, :] = dlogR_h[:t, :] + dlogmH[:t, :]  # t x num_hazard
        dlogR_m[1:t + 1, :] = dlogR_m[:t, :] + dlogpredprobs_m  # t x num_model

        dlogR_s[1:t + 1, :] = dlogR_s[:t, :] + dlogpredprobs_s  # t x num_model

        (logR[0, 0], normMsg, Z) = logsumexp(logMsg)  # 1 x 1. [P]

        # 1 x num_hazard

        dlogR_h[0, :] = rmult(dlogMsg_h, normMsg).sum(axis=0) / Z

        # 1 x num_mod

        dlogR_m[0, :] = rmult(dlogR_m[1:t + 1, :], normMsg).sum(axis=0) \
            / Z

        # 1 x num_sca

        dlogR_s[0, :] = rmult(dlogR_s[1:t + 1, :], normMsg).sum(axis=0) \
            / Z

    # end t loop

    # Get the log marginal likelihood of the data, X(1:end), under the model
    # = P(X_1:T), integrating out all the runlengths. 1 x 1. [log P]

    nlml = -1.0 * logsumexp(logR)[0]

    # Do the derivatives of nlml

    normR = np.exp(logR - max(logR))  # T x 1
    dnlml_h = -rmult(dlogR_h, normR).sum(axis=0) / sum(normR)  # 1 x num_hazard
    dnlml_m = -rmult(dlogR_m, normR).sum(axis=0) / sum(normR)  # 1 x num_model
    dnlml_s = -rmult(dlogR_s, normR).sum(axis=0) / sum(normR)  # 1 x num_scale

  # Correct for that input is log alpha0. 1 x num_scale.

    dnlml_s = alpha0 * dnlml_s

    # (num_hazard + num_model + num_scale) x 1
    dnlml = np.append(np.append(dnlml_h, dnlml_m), dnlml_s)

    assert isKosher(nlml)
    assert isKosher(dnlml)
    return (nlml, dnlml)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Utils.logit import logit

    N = 1000
    deltat = 2 * np.pi / N
    Ttrain = np.atleast_2d(range(int(.2 * N))).T * deltat
    Xtrain = np.sin(Ttrain) + 0.1 * np.random.normal(0, 1, Ttrain.shape)
    Ttest = np.atleast_2d(range(int(.2 * N), N + 1)).T * deltat
    Xtest = np.sin(Ttest) + 0.1 * np.random.normal(0, 1, Ttest.shape)

    covfunc = pyGPs.cov.RQ() + pyGPs.cov.Const() + pyGPs.cov.Noise()

    model = pyGPs.GPR()
    model.setPrior(kernel=covfunc)

    # model.setScalePrior([1.0, 1.0])

    # model.optimize(Ttrain,Xtrain)
    # model.covfunc.hyp = np.asarray([0.2803317208640337, -0.1696694454864699])

    model.covfunc.hyp = np.asarray([0.6079970164514539,
                                   0.1891319698857131,
                                   0.20521371431434787,
                                   0.016186641234573418,
                                   -2.2595732692553603])

    logtheta = model.covfunc.hyp
    theta_h = np.asarray([logit(1.0 / 50.0), 1.0, 1.0])

    num_hazard_params = theta_h.shape[0]
    if model.ScalePrior:
        theta_s = model.ScalePrior[
            0]  # alpha from the prior on scale (assumed beta is identity)
    else:
        theta_s = 0

    theta = np.append(np.append(theta_h, logtheta), theta_s)
    dt = 1

    # (nlml, dnlml) = dbocpdGP(theta, Xtest, model, num_hazard_params, dt)

    (theta_h, theta_m, theta_s, nlml) = bocpdGPTlearn(Xtrain, model,
                                                      logtheta, theta_h, dt)
    print theta_h, theta_m, theta_s, nlml
