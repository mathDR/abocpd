#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


def reverseYuleWalkerES(
    rho,
    r,
    minLen,
    epsilon,
    ):

  # function Ysub = reverseYuleWalkerES(rho, r, minLen, epsilon)
  # Solve reverse Yule Walker equations.
  # Standard Yule-Walker equations is to solve:
  #   T * y = -r  for y where T = toeplitz([rho r(1:end - 1)])
  # in 3n^2 operations as opposed to O(n^3) for general linear system. Algo found
  # in eq. 4.7.1 (on p. 195) in Sec 4.7.2 of Golub and van Loan, ref to as [GVL].
  # However, we solve the *reverse* Yule-Walker equations:
  #   T * y = r   for y where T = toeplitz([rho r(1:end - 1)])
  # Done by flipping the signs in the normal Yule-Walker algorithm.
  # Returns Ysub, which also contains solutions to sub linear systems:
  #   T * Ysub(k, 1:k) = r(1:k)  where T = toeplitz([rho r(1:k - 1)])
  # => Ysub(end, :) = y.
  # Parts of the algorithm that refer to E_k * y(1:k), multiplication by the
  # exchange matrix, are implemnted by y(k:-1:1) -- in python:  y[k::-1]
  # Note that [rho] = [r] and y is [] (unitless).

    n = r.shape[0]
    assert np.isreal(rho)
    assert r.shape == (n, 1)

  # First we must normalize r so that T has a diagonal of ones as the algorithm
  # assumes this. We don't need to correct for this at the end since the factors
  # cancel out. This makes r unitless. n x 1. []

    r = r / rho

    y = np.zeros((n, 1))
    Ysub = np.zeros((n, n))

  # Line 1 of [GVL]. 1 x 1. []

    y[0, 0] = r[0, 0]

  # Save the intermediate results for the smaller Toeplitz system. 1 x 1. []

    Ysub[0, 0] = y[0, 0]

  # Line 2 of [GVL]

    for k in range(n - 1):

    # Line 3 of [GVL]. 1 x 1. []

        beta = 1.0 - np.dot(r[:k + 1, 0].T, y[:k + 1, 0])

    # Line 4 of [GVL]. 1 x 1. []

        num = r[k + 1, 0] - np.dot(r[:k + 1, 0].T, y[k::-1, 0])
        alpha = num / beta

    # Line 5 of [GVL]. k x 1. []

        z = y[:k + 1, 0] - alpha * y[k::-1, 0]

    # Line 6 of [GVL]. k + 1 x 1. []

        y[:k + 2, 0] = np.append(z, alpha)

    # Save the intermediate results for the smaller Toeplitz system. k + 1 x 1. []

        Ysub[k + 1, :k + 2] = y[:k + 2, 0]

    # Instead of max, could use criteria of:
    # quantile(abs(alpha(ii, 1:ii - 1) - alpha(ii - 1, 1:ii - 1)), 0.95)
    # but max is much faster. If using quantile take 1/1e-2 off epsilon.

        if k >= minLen and max(np.abs(Ysub[k + 1, :k + 1] - Ysub[k, :k
                               + 1])) <= epsilon:
            break

    return Ysub


# Test routine to show equivalent to slow method

def slow_way(rho, r):
    import scipy.linalg as splinalg
    y = np.linalg.solve(splinalg.toeplitz(np.insert(r[:-1], 0, rho)), r)
    return y


def slow_way2(rho, r):

  # function Y = slow_way2(rho, r)

    n = r.shape[0]
    Y = np.zeros((n, n))

    for k in range(n):
        Y[k, :k + 1] = slow_way(rho, r[:k + 1, 0])
    return Y


if __name__ == '__main__':

    # Example taken from GVL Example 4.7.1 pp 195

    r = np.atleast_2d(np.asarray([0.5, 0.2, 0.1])).T
    rho = 1.0
    Y = slow_way2(rho, r)
    a = reverseYuleWalkerES(rho, r, 100, 1e-16)
    print 'GVL test: ', np.isclose(a, Y).all()

    # Tested on the following:

    N = 10
    r = np.atleast_2d(np.asarray(range(N + 1, 1, -1))
                      + np.random.normal(0, 1, (N, ))).T
    rho = np.random.randn()
    a = reverseYuleWalkerES(rho, r, 100, 1e-16)
    Y = slow_way2(rho, r)
    print 'Random test: ', np.isclose(Y, a).all()
