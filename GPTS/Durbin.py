#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def Durbin(r):

  # Given real numbers 1= r0, r1, ..., rn s.t.
  # T = (r_|i-j|) in R^nxn is positive definite, this algorithm
  # computes y in R^n s.t. Ty = (r1,r2,...,rn)^T
    # This is algorithm 4.7.1 from Golub and Van Loan's [GVL] Matrix Computations
  # pp. 195
  # Note the input r is assumed to be r = [r1,r2,...,rn]^T
  #

    n = r.shape[0]

    y = np.zeros_like(r)

  # Line 1 of [GVL]. 1 x 1. []

    y[0] = -r[0]
    beta = 1.0
    alpha = -r[0]

  # Line 2 of [GVL]

    for k in range(n - 1):

    # Line 3 of [GVL]. 1 x 1. []

        beta = (1 - alpha * alpha) * beta

    # Line 4 of [GVL]. 1 x 1. []

        alpha = -(r[k + 1] + np.dot(r[k::-1], y[:k + 1])) / beta

    # Line 5 of [GVL]. k x 1. []

        z = y[:k + 1] + alpha * y[k::-1]

    # Line 6 of [GVL]. k + 1 x 1. []

        y[:k + 2] = np.concatenate((z, [alpha]))
    return y


if __name__ == '__main__':
    r = np.asarray([0.5, 0.2, 0.1])  # asarray doesn't make a copy unless necessary (as opposed to array())
    y = Durbin(r)

    assert np.isclose(y, np.asarray([-75., 12., -5.]) / 140.).all()
