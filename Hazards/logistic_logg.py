#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import asarray, cumsum, zeros, zeros_like
from logistic_logh import logistic_logh

class logistic_logg():
    def __init__(self,theta):
        self.hazard_params     = theta
        self.num_hazard_params = len(theta)

    def evaluate(self,v):
        hazard = logistic_logh(self.hazard_params)
        (logH, logmH, dlogH, dlogmH) = hazard.evaluate(asarray(range(1,v+1)))

        logg  = zeros((v, ))
        dlogg = zeros_like(dlogH)

        for ii in range(v):
            if ii == 0:
                logg[ii]    = logH[ii]
                dlogg[ii,:] = dlogH[ii,:]
            else:
                logg[ii]    = logmH[:ii].sum() + logH[ii]
                dlogg[ii,:] = dlogmH[:ii,:].sum(axis=0) + dlogH[ii,:]

        # exp(logmG) = 1 - G = 1 - cumsum(g) = 1 - cumsum(exp(logg)), but this is a much
        # more numerically stable way to do it that won't underflow for G close to 1.

        logmG  = cumsum(logmH)
        dlogmG = cumsum(dlogmH, axis=0)

        return (logg, logmG, dlogg, dlogmG)

