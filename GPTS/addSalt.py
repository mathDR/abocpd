#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from copy import copy


def addSalt(Z, precision):

    # Add Gaussian white noise with mean zero, unit variance and magnitude precision to Z

    Y = copy(Z)
    if len(Y.shape) == 1:
        salt = precision * np.random.randn(Y.shape[0])
    else:
        salt = precision * np.random.randn(Y.shape[0], Y.shape[1])

    Y += salt
    return Y


if __name__ == '__main__':
    Y = np.random.randn(4, 1)
    Z = addSalt(Y, 1)
    print Y
    print Z
