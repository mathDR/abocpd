#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import asarray, argmax, exp, log


'''def logsumexp(x):
    x = asarray(x)

    maxi  = argmax(x)
    maxx = x[maxi]
    xnew = x - maxx
    xnew[maxi] = 0
    logZ     = log(exp(xnew).sum()) + maxx
    normX  = x - logZ

    return (logZ, normX)'''


def logsumexp(x):
    x = asarray(x)

    maxi = argmax(x)
    maxx = x[maxi]
    normX = exp(x - maxx)
    Z = sum(normX)
    logZ = log(Z) + maxx
    return logZ, normX, Z

if __name__ == '__main__':
    from numpy import allclose, isclose
    x = asarray([1, 2, 3])
    (a, b, c) = logsumexp(x)

    assert isclose(a, 3.40760596444)
    assert allclose(b, asarray([0.13533528, 0.36787944, 1.]))
    assert isclose(c, 1.50321472441)
    print 'logsumexp Test PASSED'
