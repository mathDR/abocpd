#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import empty
from logistic import logistic

class logistic_h2():
    def __init__(self,theta_h):
        assert (len(theta_h)==3)
        self.hazard_params = theta_h
        self.num_hazard_params = 3

    def evaluate(self,v):
        # h(t) = h * logistic(at + b)
        # theta_h: [logit(h), a, b]
        if len(v.shape) == 2:
            v = np.reshape(v, (v.shape[0], ))

        h = logistic(self.hazard_params[0])
        a = self.hazard_params[1]
        b = self.hazard_params[2]
        lp = logistic(a * v + b)
        lm = logistic(-a * v - b)
        H = h * lp
        # Derivatives
        dH = empty((v.shape[0], 3))
        dH[:, 0] = logistic(self.hazard_params[0]) * logistic(-self.hazard_params[0])*lp
        lp_lm_v = lp * lm * v
        dH[:, 1] = h * lp_lm_v
        dH[:, 2] = h * lp * lm

        return (H, dH)

