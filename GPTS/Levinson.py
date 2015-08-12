#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def Levinson(r, b):

  # Given b in R^n and real numbers 1= r0, r1, ..., rn s.t.
  # T = (r_|i-j|) in R^nxn is positive definite, this algorithm
  # computes x in R^n s.t. Tx = b
  #
  # This is algorithm 4.7.2 from Golub and Van Loan's [GVL] Matrix Computations
  # pp. 196-197

    n = r.shape[0]
    assert b.shape[0] == n

    y = np.zeros_like(r)
    x = np.zeros_like(b)

  # Line 1 of [GVL]. 1 x 1. []

    y[0] = -r[0]
    x[0] = b[0]
    beta = 1.0
    alpha = -r[0]

  # Line 2 of [GVL]

    for k in range(n - 1):

    # Line 3 of [GVL]. 1 x 1. []

        beta = (1 - alpha * alpha) * beta
        mu = (b[k + 1] - np.dot(r[:k + 1], x[k::-1])) / beta

    # Line 4 of [GVL]. 1 x 1. []

        v = x[:k + 1] + mu * y[k::-1]

    # Line 5 of [GVL]. k x 1. []

        x[:k + 2] = np.concatenate((v, [mu]))

    # Line 6 of [GVL]. k + 1 x 1. []

        if k < n - 2:
            alpha = -(r[k + 1] + np.dot(r[:k + 1], y[k::-1])) / beta  # Note there is a typo in the book!
            z = y[:k + 1] + alpha * y[k::-1]
            y[:k + 2] = np.concatenate((z, [alpha]))

    return x


if __name__ == '__main__':
    from scipy.linalg import toeplitz
    r = np.asarray([0.5, 0.2, 0.1])  # asarray doesn't make a copy unless necessary (as opposed to array())
    b = np.asarray([4.0, -1.0, 3.0])

    x = Levinson(r, b)

    assert np.isclose(x, np.asarray([355., -376., 285.]) / 56.).all()
