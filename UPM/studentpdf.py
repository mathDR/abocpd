#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gammaln, digamma


def studentpdf(x,mu,var,nu,nargout=1):
  #
  # p = studentpdf(x, mu, var, nu)
  #
  # Can be made equivalent to MATLAB's tpdf() by:
  # tpdf((y - mu) ./ sqrt(var), nu) ./ sqrt(var)
  # Equations found in p. 577 of Gelman

  # This form is taken from Kevin Murphy's lecture notes.
    c = np.exp(gammaln(nu / 2 + .5) - gammaln(nu / 2) ) * \
        (nu * np.pi * var) ** (-0.5)
    p = c * (1.0 + (1.0 / (nu * var)) * (x - mu) ** 2) ** (-0.5 * (nu + 1))

    if nargout == 2:
        if hasattr(mu, '__len__') and (not isinstance(mu, str)):
          N = len(mu)
        else:
            N = 1
        dp = np.zeros((N, 3))

        error = (x - mu) / np.sqrt(var)
        sq_error = (x - mu) ** 2 / var

        # Derivative for mu
        dlogp = (1.0 / np.sqrt(var)) * ((nu + 1.0) * error) / (nu + sq_error)

        dp[:,0] = p * dlogp

        # Derivative for var

        dlogpdprec = np.sqrt(
            var) - ((nu + 1) * (x - mu) * error) / (nu + sq_error)
        dp[:,1] = -.5 * (p / var ** 1.5) * dlogpdprec

        # Derivative for nu (df)
        dlogp = digamma(nu / 2.0 + .5) - digamma(nu / 2.0) - (1.0 / nu) - np.log(
            1.0 + (1.0 / nu) * sq_error) + ((nu + 1) * sq_error) / (nu ** 2 + nu * sq_error)
        dp[:,2] = 0.5*p* dlogp

        return (p, dp)
    return p
