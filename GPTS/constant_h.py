#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import ones_like


def constant_h(v, theta_h):
    assert theta_h >= 0
    assert theta_h <= 1

    Ht = theta_h * ones_like(v)
    dH = ones_like(v)

    return (Ht, dH)


if __name__ == '__main__':
    from numpy import asarray, allclose
    v = asarray([0.06935799, 0.63527643, 0.58383591, 0.09942945,
                0.70186987])
    theta_h = 0.10000000000000001
    (Ht, dH) = constant_h(v, theta_h)
    assert allclose(Ht, asarray([0.10000000000000001,
                    0.10000000000000001, 0.10000000000000001,
                    0.10000000000000001, 0.10000000000000001]))
    assert allclose(dH, asarray([1, 1, 1, 1, 1]))
    print 'constant_h Test PASSED'
