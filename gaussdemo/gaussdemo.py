import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gammaln

def studentpdf(x, mu, var, nu):
    # This form is taken from Kevin Murphy's lecture notes.
    c = np.exp(gammaln(0.5*(nu + 1.0)) - gammaln(0.5*nu))/np.sqrt((nu*np.pi*var))
    p = c * (1 + (1.0/(nu*var))*(x-mu)**2)**(-(nu+1)/2)
    return p


def constant_hazard(r, lam):
  return 1.0/lam * np.ones_like(r)


if __name__ == '__main__':

    # How many time steps to generate?
    T = 1000

    # Specify the hazard function.
    # This is a handle to a function that takes one argument - the number of
    # time increments since the last changepoint - and returns a value in
    # the interval [0,1] that is the probability of changepoint.  Generally
    # you might want to have your hazard function take parameters, so using
    # an anonymous function is helpful.  We're going to just use the simple
    # constant-rate hazard function that gives geomtrically-drawn intervals
    # between changepoints.  We'll specify the rate via a mean.
    lam        = 200
    hazard_func  = lambda r: constant_hazard(r, lam)

    # This data is Gaussian with unknown mean and variance.  We are going to
    # use the standard conjugate prior of a normal-inverse-gamma.  Note that
    # one cannot use non-informative priors for changepoint detection in
    # this construction.  The NIG yields a closed-form predictive
    # distribution, which makes it easy to use in this context.  There are
    # lots of references out there for doing this kind of inference - for
    # example Chris Bishop's "Pattern Recognition and Machine Learning" in
    # Chapter 2.  Also, Kevin Murphy's lecture notes.
    mu0    = 0
    kappa0 = 1
    alpha0 = 1
    beta0  = 1

    # This will hold the data.  Preallocate for a slight speed improvement.
    X = []

    # Store the times of changepoints.  It's useful to see them.
    CP = []

    # Generate the initial parameters of the Gaussian from the prior.
    curr_ivar = np.random.gamma(alpha0)/beta0
    curr_mean = np.random.randn()/np.sqrt((kappa0*curr_ivar)) + mu0

    # The initial run length is zero.
    curr_run = 0

    # Now, loop forward in time and generate data.
    for t in range(T):

      # Get the probability of a new changepoint.
      p = hazard_func(curr_run)

      # Randomly generate a changepoint, perhaps.
      if np.random.rand() < p:

        # Generate new Gaussian parameters from the prior.
        curr_ivar = np.random.gamma(alpha0)*beta0
        curr_mean = np.random.randn()/np.sqrt(kappa0*curr_ivar) + mu0

        # The run length drops back to zero.
        curr_run = 0

        # Add this changepoint to the end of the list.
        CP.append(t)

      else:

        # Increment the run length if there was no changepoint.
        curr_run += 1

        # Draw data from the current parameters.
      X.append( np.random.randn()/np.sqrt(curr_ivar) + curr_mean )

    # Plot the data and we'll have a look.
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.plot(range(T), X, 'b-')
    for c in CP:
      ax.plot([c,c], [1.5*np.min(X),1.5*np.max(X)], 'rx-')
    ax.grid


    # Now we have some data in X and it's time to perform inference.
    # First, setup the matrix that will hold our beliefs about the current
    # run lengths.  We'll initialize it all to zero at first.  Obviously
    # we're assuming here that we know how long we're going to do the
    # inference.  You can imagine other data structures that don't make that
    # assumption (e.g. linked lists).  We're doing this because it's easy.
    R = 1e-16*np.ones((T+1, T))

    # At time t=0, we actually have complete knowledge about the run
    # length.  It is definitely zero.  See the paper for other possible
    # boundary conditions.
    R[0,0] = 1

    # Track the current set of parameters.  These start out at the prior and
    # accumulate data as we proceed.
    muT    = mu0
    kappaT = kappa0
    alphaT = alpha0
    betaT  = beta0

    # Keep track of the maximums.
    maxes  = np.zeros((T+1,))

    # Loop over the data like we're seeing it all for the first time.
    for t in range(1,T):

      # Evaluate the predictive distribution for the new datum under each of
      # the parameters.  This is the standard thing from Bayesian inference.
      predprobs = studentpdf(X[t], muT,betaT*(kappaT+1)/(alphaT*kappaT),2.0*alphaT)

      # Evaluate the hazard function for this interval.
      H = hazard_func(np.asarray(range(t)))

      # Evaluate the growth probabilities - shift the probabilities down and to
      # the right, scaled by the hazard function and the predictive
      # probabilities.

      R[1:t+1,t] = R[:t,t-1]*predprobs*(1-H)

      # Evaluate the probability that there *was* a changepoint and we're
      # accumulating the mass back down at r = 0.
      R[0,t] = np.sum( R[:t,t-1]*predprobs*H )

      # Renormalize the run length probabilities for improved numerical
      # stability.
      R[:,t] = R[:,t] / (1e-16 + np.sum(R[:,t]))

      # Update the parameter sets for each possible run length.
      muT0    = np.append( mu0    , (kappaT*muT + X[t]) / (kappaT+1) )
      kappaT0 = np.append( kappa0 , kappaT + 1. )
      alphaT0 = np.append( alpha0 , alphaT + 0.5 )
      betaT0  = np.append( beta0  , betaT + (kappaT *(X[t]-muT)**2)/(2.0*(kappaT+1)) )
      muT     = muT0
      kappaT  = kappaT0
      alphaT  = alphaT0
      betaT   = betaT0

      # Store the maximum, to plot later.
      maxes[t] = np.argmax(R[:,t])


    # Show the log smears and the maximums.
    ax2 = fig.add_subplot(2,1,2, sharex = ax)
    #ax2.imshow(-np.log(R), cmap = plt.get_cmap('gray'), origin='lower')

    ax2.plot(range(1,T+2), maxes, 'r-')
    ax2.set_xlim([0, T])
    ax2.set_ylim([0, T])
    plt.show()



