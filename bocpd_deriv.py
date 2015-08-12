import numpy as np

def bocpd_deriv(theta_h, theta_m, X, hazard_f, model_f):
  num_hazard_params = len(theta_h)
  num_model_params  = len(theta_m)

  if len(X.shape) == 2:
    T,D = X.shape # Number of time point observed. 1 x 1. [s]
  else:
    T = X.shape[0]
    D = 1

  # R(r, t) = P(runlength_t-1 = r-1|X_1:t-1).
  R    = np.zeros((T + 1, T + 1)) # pre-load the run length distribution. [P]
  dR_h = np.zeros((T + 1, T + 1, num_hazard_params))
  dR_m = np.zeros((T + 1, T + 1, num_model_params))

  # At time t = 1, we actually have complete knowledge about the run
  # length.  It is definitely zero.  See the paper for other possible
  # boundary conditions. This assumes there was surely a change point right
  # before the first data point not at the first data point.
  # Implements step 1, alg 1, of [RPA].
  # => P(runglenth_0 = 0|nothing) = 1
  R[0,0] = 1 # 1 x 1. [P]

  # The evidence at each time step => Z(t) = P(X_t|X_1:t-1). [P]
  Z    = np.zeros((T, 1))
  dZ_h = np.zeros((T, num_hazard_params))
  dZ_m = np.zeros((T, num_model_params))
  # Find parameters of p(X_1|nothing). 1 x param_count. units depend on model_f
  model_f.init_f(T + 1, D, theta_m)
  for t in range(T):
    # Implictly Implements step 2, alg 1, of [RPA]: oberserve new datum, simply by
    # incrementing the loop index.

    # Evaluate the predictive distribution for the new datum under each of
    # the parameters.  This is the standard thing from Bayesian inference.
    # Implements step 3, alg 1, of [RPA].
    # predprobs(r) = p(X(t)|X(1:t-1), runlength_t-1 = r-1). t x 1. [P]
    predprobs, dpredprobs = model_f.predict(model_f.post_params, X[t,:], model_f.dmessage)

    # Evaluate the hazard function for this interval.
    # H(r) = P(runlength_t = 0|runlength_t-1 = r-1)
    # Pre-computed the hazard in preperation for steps 4 & 5, alg 1, of [RPA]
    H, dH = hazard_f.evaluate(range(t)) # t x 1. [P]

    # Evaluate the growth probabilities - shift the probabilities up and to
    # the right, scaled by the hazard function and the predictive
    # probabilities.
    # Implements step 4, alg 1, of [RPA].
    # Assigning P(runlength_t = 1|X_1:t) to P(runlength_t = t|X_1:t):
    # P(runlength_t = r|X_1:t) propto P(runlength_t-1 = r-1|X_1:t-1) *
    # p(X_t|X_1:t-1,runlength_t-1 = r-1) * P(runlength_t = r|runlength_t-1 = r-1).
    R[1:t + 1, t + 1] = R[:t, t]*predprobs*(1 - H) # t x 1. [P]
    for ii in range(hazard_f.num_hazard_params):
      dR_h[1:t + 1, t + 1, ii] = predprobs*(dR_h[:t, t, ii]*(1 - H)-R[:t, t]*dH[:,ii])

    for ii in range(num_model_params):
      dR_m[1:t + 1, t + 1, ii] = (1 - H)*(dR_m[:t,t,ii]*predprobs+ R[:t,t]* dpredprobs[:,ii])

    # Evaluate the probability that there *was* a changepoint and we're
    # accumulating the mass back down at r = 0.
    # Implements step 5, alg 1, of [RPA].
    # Assigning P(runlength_t = 0|X_1:t)
    # P(runlength_t = 0|X_1:t) propto sum_r=0^t-1 P(runlength_t-1 = r|X_1:t-1) *
    # p(X_t|X_1:t-1, runlength_t-1 = r) * P(runlength_t = 0|runlength_t-1 = r).
    R[0, t + 1] = np.sum(R[:t, t] * predprobs * H) # 1 x 1. [P]
    for ii in range(hazard_f.num_hazard_params):
      dR_h[0, t + 1, ii] = np.sum(predprobs * (dR_h[:t, t, ii] * H + R[:t,t] * dH[:, ii]))
    for ii in range(model_f.num_model_params):
      dR_m[0, t + 1, ii] = np.sum(H * (dR_m[:t, t, ii] * predprobs + R[:t,t] * dpredprobs[:, ii]))


    # Renormalize the run length probabilities for improved numerical
    # stability.
    # note that unlike in [RPA] which keeps track of P(r_t, X_1:t), we keep track
    # of P(r_t|X_1:t) => unnormalized R(i, t+1) = P(runlength_t = i-1|X_1:t) *
    # P(X_t|X_1:t-1) => normalization const Z(t) = P(X_t|X_1:t-1).
    # Sort of Implements step 6, alg 1, of [RPA].
    # Could sum R(:, t+1), but we only sum R(1:t+1,t+1) to avoid wasting time
    # summing zeros.
    Z[t] =  np.sum(R[:t + 1, t + 1]) # 1 x 1. [P]
    for ii in range(num_hazard_params):
      dZ_h[t, ii] = np.sum(dR_h[:t + 1, t + 1, ii])
    for ii in range(num_model_params):
      dZ_m[t, ii] = np.sum(dR_m[:t + 1, t + 1, ii])
    # Implements step 7, alg 1, of [RPA].
    # After normalization, R(i, t+1) = P(runlength_t = i-1|X_1:t).
    for ii in range(num_hazard_params):
      dR_h[:t + 1, t + 1, ii] = (dR_h[:t + 1, t + 1, ii]/Z[t]) - (dZ_h[t, ii] * R[:t + 1, t + 1]) / (Z[t] ** 2)
    for ii in range(num_model_params):
      dR_m[:t + 1, t + 1, ii] = (dR_m[:t + 1, t + 1, ii] / Z[t]) - (dZ_m[t, ii] * R[:t + 1, t + 1]) / (Z[t] ** 2)
    R[:t + 1, t + 1] = R[:t + 1, t + 1] /  Z[t] # (T + 1) x 1. [P]

    # post_params(r, i) = current state of hyper-parameter i given runlength = r-1
    # Implements step 8, alg 1, of [RPA].
    # (t + 1) x param_count. units depend on model_f.
    post_params, dmessage = model_f.update(theta_m, post_params, X[t, :], dmessage)

  # Could implement step 9, alg 1, of [RPA] is already taken care of by Z(t).


  # Get the log marginal likelihood of the data, X(1:end), under the model
  # = P(X_1:T), integrating out all the runlengths. 1 x 1. [log P]
  nlml    = -np.sum(np.log(Z))
  dnlml_h = np.zeros((num_hazard_params, 1))
  dnlml_m = np.zeros((num_model_params, 1))
  for ii in range(num_hazard_params):
    dnlml_h[ii] = -np.sum(dZ_h[:, ii]/Z)
  for ii in range(num_model_params):
    dnlml_m[ii] = -np.sum(dZ_m[:, ii]/Z)

  return nlml, dnlml_h, dnlml_m, Z, dZ_h, dZ_m, R, dR_h, dR_m
