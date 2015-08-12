#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


def TIMpredict(N, mu, sig):

    # Return N draws from a normal distribution with mean mu and variance sig

    return np.random.normal(mu, sig, N)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from numpy import linspace
    x = linspace(-1.0, 1.0, 200)
    z = TIMpredict(len(x), 0.5, .05)
    plt.plot(x, z, 'r.')
    plt.plot(x, z, 'b')
    plt.show()
