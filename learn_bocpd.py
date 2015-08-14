import numpy as np

from rt_minimize import rt_minimize
from Hazards import logistic

def learn_bocpd(X, model_f, hazard_f, conversion):
  max_minimize_iter = 30

  theta = hazard_f.hazard_params
  theta.extend(model_f.post_params)

  theta, nlml, _ = rt_minimize(theta, bocpd_dwrap1D, -max_minimize_iter, X, model_f, hazard_f, conversion, hazard_f.num_hazard_params)
  print "we make it to here!"

  hazard_f.hazard_params = theta[0:num_hazard_params]
  model_f.model_params   = theta[num_hazard_params:]

  hazard_f.hazard_params[0] = logistic(hazard_f.hazard_params[0])
  model_f.model_params[1:]  = np.exp(model_f.model_params[1:])
  return hazard_params, model_params, nlml

def bocpd_dwrap1D(theta, X, model_f, hazard_f, conversion):
  from bocpd_deriv import bocpd_deriv
  # Warning: this code assumes: theta_h are in logit scale, theta_m(1) is in
  # linear, and theta_m(2:end) are in log scale!

  theta[conversion == 2] = logistic(theta[conversion == 2])
  theta[conversion == 1] = np.exp(theta[conversion == 1])

  # Seperate theta into hazard and model hypers
  theta_h = theta[:hazard_f.num_hazard_params]
  theta_m = theta[hazard_f.num_hazard_params:]

  nlml, dnlml_h, dnlml_m = bocpd_deriv(theta_h, theta_m, X, hazard_f, model_f)

  # Put back into one vector for minimize
  dnlml = dnlml_h.extend(dnlml_m)

  # Correct for the distortion
  dnlml[conversion == 2] = dnlml[conversion == 2] * theta[conversion == 2] * (1 - theta[conversion == 2])
  dnlml[conversion == 1] = dnlml[conversion == 1] * theta[conversion == 1]

  return nlml, dnlml
