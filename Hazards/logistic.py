#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import exp

def logistic(x):
    return 1. / (1. + exp(-x))

