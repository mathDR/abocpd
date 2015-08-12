# Ryan Turner (rt324@cam.ac.uk)
# Evaluate the BOCPS learning procedure on the well data and compare to RPA hand
# picked hyper-parameters.
import numpy as np
from scipy.special import logit
import matplotlib.pyplot as plt
from time import clock
import random

from learn_bocpd import learn_bocpd
from Hazards import logistic
from Hazards import constant_h
from Hazards import logistic_h
from UPM import gaussian1D

if __name__ == '__main__':
  print 'Trying well log data'

  # I think all this code is deterministic, but let's fix the seed to be safe.
  random.seed(4)

  well = np.genfromtxt('Data/well.dat')

  # We don't know the physical interpretation, so lets just standardize the
  # readings to make them cleaner.
  X = (well - np.mean(well,axis=0))/(1e-16 + np.std(well,axis=0))
  Tlearn = 2000
  Ttest  = X.shape[0] - Tlearn

  useLogistic = True


  model_init = np.asarray([0, np.log(.1), np.log(.1), np.log(.1)])
  model_f = gaussian1D()
  model_f.init_f(1,1,model_init)
  if useLogistic:
    hazard_init = [logit(.01), 0, 0]
    hazard_f = logistic_h(hazard_init)
    conversion = [2, 0, 0, 0, 1, 1, 1]
  else:
    hazard_f = constant_h(logit(.01))
    conversion = [2, 0, 1, 1, 1]
  print X.shape, well.shape, Tlearn, Ttest, useLogistic
  #assert(isVector(X))
  #assert(X, 2) == 1)

  # TODO compare logistic h and constant h

  # Can try learn_IFM usinf IFMfast to speed this up
  print 'Starting learning'
  start_time = clock()
  well_hazard, well_model, well_learning = learn_bocpd(X[:Tlearn], model_f,hazard_f, conversion)
  print clock() - start_time
  print 'Learning Done'

  '''print 'Testing'
  start_time = clock()
  [well_R, well_S, well_nlml, Z] = bocpd(X, 'gaussian1D', well_model', 'logistic_h', well_hazard')
  print clock() - start_time
  print 'Done Testing'

  nlml_score = -np.sum(np.log(Z(Tlearn + 1:end))) / Ttest

  # TODO assert we get the same results as [RPA] without doing standardizing
  # from the paper sec 3.1
  rpa_hazard   = 1 / 250
  rpa_mu0      = 1.15e5
  rpa_mu_sigma = 1e4

  # correct for the effects of standardizing
  rpa_mu0 = (rpa_mu0 - mean(well)) / std(well)
  rpa_mu_sigma = rpa_mu_sigma / std(well)

  # convert to precision
  rpa_kappa = 1 / rpa_mu_sigma ^ 2

  # unstated what he uses for variance parameters.
  # some reasonable defaults
  rpa_alpha = 1
  rpa_beta = rpa_kappa

  rpa_model = [rpa_mu0 1 rpa_alpha rpa_beta]
  [well_R, well_S_rpa, well_nlml_rpa, Z_rpa] = bocpd(X, 'gaussian1D', rpa_model, 'constant_h', rpa_hazard)

  nlml_score_rpa = -sum(log(Z_rpa(Tlearn + 1:end))) / Ttest

  TIM_nlml = -sum(normlogpdf(X(Tlearn + 1:end))) / Ttest

  well_results

  plotS(well_S, X)
  title(['RDT ' num2str(nlml_score)])

  plotS(well_S_rpa, X)
  title(['RPA ' num2str(nlml_score_rpa)])
  '''
