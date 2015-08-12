#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import exp


def logistic(x):
    return 1. / (1. + exp(-x))


if __name__ == '__main__':
    from numpy import asarray, allclose
    x = asarray([1, 23, 3])
    assert allclose(logistic(x), asarray([0.73105858, 1., 0.95257413]))
    print 'logistic Test PASSED'
