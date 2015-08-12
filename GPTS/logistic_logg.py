#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from logistic_logh import logistic_logh


def logistic_logg(v, theta):
    (logH, logmH, dlogH, dlogmH) = logistic_logh(np.asarray(range(1, v
            + 1)), theta)

    logg = np.zeros((v, ))
    dlogg = np.zeros_like(dlogH)

    for ii in range(v):
        if ii == 0:
            logg[ii] = logH[ii]
            dlogg[ii, :] = dlogH[ii, :]
        else:
            logg[ii] = logmH[:ii].sum() + logH[ii]
            dlogg[ii, :] = dlogmH[:ii, :].sum(axis=0) + dlogH[ii, :]

    # exp(logmG) = 1 - G = 1 - cumsum(g) = 1 - cumsum(exp(logg)), but this is a much
    # more numerically stable way to do it that won't underflow for G close to 1.

    logmG = np.cumsum(logmH)
    dlogmG = np.cumsum(dlogmH, axis=0)

    return (logg, logmG, dlogg, dlogmG)


if __name__ == '__main__':
    from numpy import asarray, allclose
    v = 4
    theta_h = asarray([1.5, 2.5, .5])
    (logg, logmG, dlogg, dlogmG) = logistic_logg(v, theta_h)

    assert allclose(logg, asarray([-0.25000063, -1.71418105,
                    -3.39377458, -5.09337818]))
    assert allclose(logmG, asarray([-1.50868933, -3.19202589,
                    -4.89193736, -6.59322724]))
    assert allclose(dlogg, asarray([[0.18242552, 0.04742587,
                    0.04742587], [-0.45985856, -0.15883682,
                    -0.16290695], [-1.25951872, -0.20165384,
                    -0.18448314], [-2.07559252, -0.20705028,
                    -0.18629113]]))
    assert allclose(dlogmG, asarray([[-0.64228408, -0.16697709,
                    -0.16697709], [-1.44194425, -0.20265989,
                    -0.18481849], [-2.25801804, -0.20716042,
                    -0.18631867], [-3.07546913, -0.20765398,
                    -0.18644206]]))
    print 'logistic_logg Test PASSED'
