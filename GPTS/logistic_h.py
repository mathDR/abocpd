#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from logistic import *


def logistic_h(v, theta_h):

    # h(t) = h * logistic(at + b)
    # theta_h: [h, a, b]

    if len(v.shape) == 2:
        v = np.reshape(v, (v.shape[0], ))

    h = theta_h[0]
    a = theta_h[1]
    b = theta_h[2]

    lp = logistic(a * v + b)

    lm = logistic(-a * v - b)
    H = h * lp

    # Derivatives

    dH = np.empty((v.shape[0], 3))
    dH[:, 0] = lp
    lp_lm_v = lp * lm * v
    dH[:, 1] = h * lp_lm_v
    dH[:, 2] = h * lp * lm

    return (np.atleast_2d(H).T, dH)


if __name__ == '__main__':
    v = np.random.random((3, 1))
    theta_h = np.asarray([0.1, 1, 1])
    print logistic_h(v, theta_h)
