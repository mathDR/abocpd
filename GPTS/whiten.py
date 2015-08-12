#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


def whiten(x, N=None):
    minLambda = 1e-10

    T = x.shape[0]

    if N == None:
        N = T

    x_bar = np.atleast_2d(np.mean(x[:N, :], axis=0))
    sigma = np.cov(x[:N, :])
    print sigma.shape

   # eigenvalue decomposition of the covariance matrix

    (lam, U) = np.linalg.eigh(sigma)

   # print np.dot(U,np.dot(np.diag(lam),U.T)) - sigma

   # Avoid numerical problems

    lam = np.diag(lam)
    lam[lam < minLambda] = minLambda
    lamInverse = 1 / np.sqrt(lam)

    print np.dot(U.T, np.dot(lamInverse, U).shape)

   # return np.dot(U.T, np.dot(np.dot(lamInverse,U),(x-np.dot(np.ones((T,1)),x_bar))) ).T

    return 0


if __name__ == '__main__':
    Y = np.random.random((4, 3))

   # print np.cov(Y)

    Z = whiten(Y)

   # print np.cov(Z)
