#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from logistic import *


def logistic_h2(v, theta_h):

    # h(t) = h * logistic(at + b)
    # theta_h: [logit(h), a, b]

    if len(v.shape) == 2:
        v = np.reshape(v, (v.shape[0], ))

    h = logistic(theta_h[0])
    a = theta_h[1]
    b = theta_h[2]

    lp = logistic(a * v + b)
    lm = logistic(-a * v - b)
    H = h * lp

    # Derivatives

    dH = np.empty((v.shape[0], 3))
    dH[:, 0] = logistic(theta_h[0]) * logistic(-theta_h[0]) * lp

    lp_lm_v = lp * lm * v

    dH[:, 1] = h * lp_lm_v
    dH[:, 2] = h * lp * lm

    return (H, dH)


if __name__ == '__main__':
    from numpy import asarray, allclose
    v = asarray([0.44666091, 0.44823091, 0.38885706])
    theta_h = asarray([0.1, 1, 1])
    (a, b) = logistic_h2(v, theta_h)
    assert allclose(a, asarray([0.42496226, 0.42508931, 0.42019844]))
    assert allclose(b, asarray([[0.20186592, 0.03616261, 0.0809621],
                    [0.20192627, 0.03625446, 0.08088343], [0.19960301,
                    0.03261248, 0.08386753]]))
    print 'logistic_h2 Test PASSED'
