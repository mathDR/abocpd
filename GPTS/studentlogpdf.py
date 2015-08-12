#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gammaln, digamma


def studentlogpdf(
    x,
    mu,
    var,
    nu,
    nargout=1,
    ):

  #
  # p = studentpdf(x, mu, var, nu)
  #
  # Can be made equivalent to MATLAB's tpdf() by:
  # tpdf((y - mu) ./ sqrt(var), nu) ./ sqrt(var)
  # Equations found in p. 577 of Gelman

    computeDerivs = nargout == 2

    error = (x - mu) / np.sqrt(var)
    sq_error = (x - mu) ** 2 / var

  # This form is taken from Kevin Murphy's lecture notes.

    c = gammaln(nu / 2 + .5) - gammaln(nu / 2) - .5 * np.log(nu * np.pi
            * var)
    p = c + -(nu + 1) / 2 * np.log(1 + sq_error / nu)

    if computeDerivs:
        N = len(mu)
        dp = np.zeros((N, 3))

    # Derivative for mu

        dp[:, 0] = 1 / np.sqrt(var) * ((nu + 1) * error) / (nu + sq_error)

    # Derivative for var

        dlogpdprec = np.sqrt(var) - (nu + 1) * np.sqrt(var) * sq_error \
            / (nu + sq_error)
        dp[:, 1] = -.5 * (1 / var ** 1.5) * dlogpdprec

    # Derivative for nu (df)

        dlogp = digamma(nu / 2 + .5) - digamma(nu / 2) - 1 / nu \
            - np.log(1 + 1 / nu * sq_error) + (nu + 1) * sq_error / (nu
                ** 2 + nu * sq_error)
        dp[:, 2] = .5 * dlogp

        return (p, dp)
    return p


if __name__ == '__main__':
    mu = np.random.normal(0, 1, (4, ))
    var = np.random.random((4, ))
    nu = np.random.random((4, ))
    x = np.random.normal(0, 1, (4, ))
    (p, dp) = studentlogpdf(x, mu, var, nu, nargout=2)
    print p, dp
