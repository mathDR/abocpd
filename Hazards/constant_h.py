#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import ones_like

class constant_h():
    def __init__(self,theta_h):
        if isinstance(theta_h, list):
            assert (len(theta_h)==1)
            theta_h = theta_h[0]

        assert theta_h >= 0
        assert theta_h <= 1
        self.hazard_params = theta_h
        self.num_hazard_params = 1

    def evaluate(self,v):
        Ht = self.hazard_params * ones_like(v)
        dH = ones_like(v)
        return (Ht, dH)



