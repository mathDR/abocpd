#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gammaln, digamma


def studentpdf(
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

  # This form is taken from Kevin Murphy's lecture notes.
    c = np.exp(gammaln(nu / 2 + .5) - gammaln(nu / 2) ) * \
        (nu * np.pi * var) ** (-0.5)
    p = c * (1.0 + (1.0 / (nu * var)) * (x - mu) ** 2) ** (-0.5 * (nu + 1))

    if nargout == 2:
        N = len(mu)
        dp = np.zeros((N, 3))

        error = (x - mu) / np.sqrt(var)
        sq_error = (x - mu) ** 2 / var

        # Derivative for mu
        dlogp = (1.0 / np.sqrt(var)) * ((nu + 1.0) * error) / (nu + sq_error)

        dp[:, 0] = p * dlogp

        # Derivative for var

        dlogpdprec = np.sqrt(
            var) - ((nu + 1) * (x - mu) * error) / (nu + sq_error)
        dp[:, 1] = -.5 * (p / var ** 1.5) * dlogpdprec

        # Derivative for nu (df)
        dlogp = digamma(nu / 2.0 + .5) - digamma(nu / 2.0) - (1.0 / nu) - np.log(
            1.0 + (1.0 / nu) * sq_error) + ((nu + 1) * sq_error) / (nu ** 2 + nu * sq_error)
        dp[:, 2] = .5 * p * dlogp

        return (p, dp)
    return p


if __name__ == '__main__':
    x = np.asarray([0.0608528,   0.1296728,  -0.2238741,   0.79862108])
    mu = np.asarray([-0.85759774,  0.70178911, -0.29351646,  1.60215909])
    var = np.asarray([0.82608497,  0.75882319,  0.86101641,  0.73113357])
    nu = np.asarray([0.71341641,  0.52532607,  0.20685246,  0.02304925])

    p = studentpdf(x, mu, var, nu, nargout=1)
    assert np.allclose(
        p, np.asarray([0.1521209,   0.1987373,   0.21214484,  0.01335992]))

    (p, dp) = studentpdf(x, mu, var, nu, nargout=2)
    assert np.allclose(
        p, np.asarray([0.1521209,   0.1987373,   0.21214484,  0.01335992]))
    assert np.allclose(
        dp, np.asarray([[1.67068098e-01,   8.00695192e-04,   9.07088043e-02],
                        [-2.38903047e-01,  -4.08902709e-02,   1.76043126e-01],
                        [9.74584714e-02,  -1.19253012e-01,   4.08675818e-01],
                        [-1.65769327e-02,  -2.71641034e-05,   5.45223728e-01]]))
    print 'studentpdf Test PASSED'
