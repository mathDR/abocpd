#!/usr/bin/python
# -*- coding: utf-8 -*-
# Ryan Turner (rt324@cam.ac.uk)
# Checks to see if a matrix is kosher meaning all the values are finite (no
# -Inf, +Inf, NaN) and it is real (no complex numbers).  If check_pos is 1 then
# it also checks if all the elements are more than zero.

from numpy import isfinite, isreal


def isKosher(X, check_pos=None):
    if check_pos == 1:
        kosher = isfinite(X).all() and isreal(X) and (X > 0).all()
    else:
        kosher = isfinite(X).all() and isreal(X).all()
    return kosher


if __name__ == '__main__':
    import numpy as np
    X = np.random.random((3, 1))
    assert isKosher(X)
    X[0] = np.nan
    assert not isKosher(X)
    X[0] = np.inf
    assert not isKosher(X)
    print 'isKosher test PASSED'
