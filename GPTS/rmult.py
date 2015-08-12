#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import atleast_2d, tile


def rmult(A, b):

  # row multiplication: Z = A*b row-wise
  # b must have one column

    assert len(A.shape) == 2
    if len(b.shape) == 1:
        b = atleast_2d(b).T

    (N, D) = A.shape
    (K, q) = b.shape

    if N != K or q != 1:
        raise Error('Error in rMult')

    return A * tile(b, D)


if __name__ == '__main__':
    import numpy as np
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([1, 2, 3])
    assert np.allclose(rmult(A, b), np.array([[1, 2, 3], [8, 10, 12],
                       [21, 24, 27]]))
    print 'rmult test PASSED'
