#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import log


def logit(p):
    return log(p) - log(1 - p)


if __name__ == '__main__':
    import numpy as np
    p = np.asarray([0.74998196, 0.68692041, 0.09972072, 0.86273085,
                   0.53687572])
    if np.allclose(logit(p), np.asarray([1.09851609, 0.78576098,
                   -2.20033153, 1.83815917, 0.1477712])):
        print 'logit Test PASSED'
    else:
        print 'logit Test FAILED'
