import numpy as np
from scipy.special import logit

from Hazards import logistic
from Hazards import constant_h
from Hazards import logistic_h
from UPM import gaussian1D
from rt_minimize import rt_minimize

def learn_bocpd(X, useLogistic=False):
  max_minimize_iter = 30

  if useLogistic:
    model_f = gaussian1D()
    hazard_f = logistic_h
    num_hazard_params = 3

    hazard_init = [logit(.01), 0, 0]
    model_init = [0, np.log(.1), np.log(.1), np.log(.1)]
    conversion = [2, 0, 0, 0, 1, 1, 1]
  else:
    model_f = 'gaussian1D'
    hazard_f = constant_h
    num_hazard_params = 1

    hazard_init = [logit(.01)]
    model_init = [0, np.log(.1), np.log(.1), np.log(.1)]
    conversion = [2, 0, 1, 1, 1]

  theta = hazard_init.extend(model_init)

  theta, nlml = rt_minimize(theta, bocpd_dwrap1D, -max_minimize_iter, X, model_f, hazard_f, conversion, num_hazard_params)

  hazard_params = theta[0:num_hazard_params]
  model_params  = theta[num_hazard_params:]

  hazard_params[0] = logistic(hazard_params[0])
  model_params[1:] = np.exp(model_params[1:])
  return hazard_params, model_params, nlml

def bocpd_dwrap1D(theta, X, model_f, hazard_f, conversion, num_hazard_params):
  from bocpd_deriv import bocpd_deriv
  # Warning: this code assumes: theta_h are in logit scale, theta_m(1) is in
  # linear, and theta_m(2:end) are in log scale!

  theta[conversion == 2] = logistic(theta[conversion == 2])
  theta[conversion == 1] = np.exp(theta[conversion == 1])

  # Seperate theta into hazard and model hypers
  theta_h = theta[:num_hazard_params]
  theta_m = theta[num_hazard_params:]

  nlml, dnlml_h, dnlml_m = bocpd_deriv(theta_h, theta_m, X, hazard_f, model_f)

  # Put back into one vector for minimize
  dnlml = dnlml_h.extend(dnlml_m)

  # Correct for the distortion
  dnlml[conversion == 2] = dnlml[conversion == 2] * theta[conversion == 2] * (1 - theta[conversion == 2])
  dnlml[conversion == 1] = dnlml[conversion == 1] * theta[conversion == 1]

  return nlml, dnlml
