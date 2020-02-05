#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from logistic import *

class logistic_logh():
    def __init__(self,theta_h):
        self.hazard_params = theta_h
        self.num_hazard_params = len(theta_h)


    def logsumexp(self,x, c):
        # function logZ = logsumexp(x, c)
        maxx = np.max([np.max(x),np.max(c)])
        if isinstance(maxx, np.ndarray):
            maxx[np.isnan(maxx)] = 0
            maxx[np.logical_not(np.isfinite(maxx))] = 0
        return np.log(np.exp(x - maxx) + np.exp(c - maxx)) + maxx


    def dlogsumexp(self,x,dx,c,dc):
        maxx = np.max([np.max(x), np.max(c)])
        if isinstance(maxx, np.ndarray):
            maxx[np.isnan(maxx)] = 0
            maxx[np.logical_not(np.isfinite(maxx))] = 0

        # TODO rewrite to avoid 0/0 in extreme cases

        return (np.exp(x - maxx) * dx + np.exp(c - maxx) * dc) / (np.exp(x- maxx) + np.exp(c - maxx))


    def loglogistic(self,x):
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

    def evaluate(self,v):
        # h(t) = h * logistic(at + b)
        # theta_h: [logit(h), a, b]
        # derived on p. 230 - 232 of DPI notebook

        if np.isscalar(v):
            v = np.asarray([v])
        T = v.shape[0]
        if len(v.shape) == 2:
            v = np.reshape(v, (T, ))

        (h, a, b) = (logistic(self.hazard_params[0]), self.hazard_params[1], self.hazard_params[2])

        logmh = self.loglogistic(-self.hazard_params[0])
        logh  = self.loglogistic(self.hazard_params[0])

        logisticNeg = logistic(-a*v - b)

        logH = self.loglogistic(a * v + b) + logh

        logmH = self.logsumexp(-a * v - b, logmh) + self.loglogistic(a * v + b)

        if len(logH.shape) == 1:
            logH = np.atleast_2d(logH).T

        if len(logmH.shape) == 1:
            logmH = np.atleast_2d(logmH).T

        # Derivatives

        dlogH      = np.zeros((T, 3))
        dlogH[:,0] = logistic(-self.hazard_params[0])
        dlogH[:,1] = logisticNeg*v
        dlogH[:,2] = logisticNeg

        dlogmH = np.zeros((T, 3))
        dlogmH[:,0] = self.dlogsumexp(-a * v - b, 0, logmh, -h)
        dlogmH[:,1] = self.dlogsumexp(-a * v - b, -v, logmh, 0) + v*logisticNeg
        dlogmH[:,2] = self.dlogsumexp(-a * v - b, -1, logmh, 0) + logisticNeg

        assert logH.shape   == (T, 1)
        assert logmH.shape  == (T, 1)
        assert dlogH.shape  == (T, 3)
        assert dlogmH.shape == (T, 3)

        return (logH, logmH, dlogH, dlogmH)


