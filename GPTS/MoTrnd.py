#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from copy import copy


def MoTrnd(
    w1,
    mus,
    var,
    df,
    samples,
):
    w = copy(w1)
    ww = w[0]
    w[w <= 1e-6] = 0
    if ww > 1e-6:
        w[0] = ww

    probs = np.random.multinomial(1, w, size=samples)
    idx = [list(k).index(1) for k in probs]
    X = np.sqrt(np.asarray(var)[idx]) * np.random.standard_t(
        np.asarray(df)[idx], samples) + np.asarray(mus)[idx]
    return np.asarray(X)


if __name__ == '__main__':
    from time import clock

    w1 = [0.1, 0.2, 0.7]
    mus = [0, 0, 0]
    var = [0.5, 0.5, 0.5]
    df = [4., 4., 4.]
    samples = 1000

    X = MoTrnd(w1, mus, var, df, samples)
    print np.median(X)
