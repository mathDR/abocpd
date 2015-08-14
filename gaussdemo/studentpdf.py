import numpy as np
from scipy.special import gammaln

def studentpdf(x, mu, var, nu):
    # This form is taken from Kevin Murphy's lecture notes.
    c = np.exp(gammaln(0.5*(nu + 1.0)) - gammaln(0.5*nu))/np.sqrt((nu*np.pi*var))
    p = c * (1 + (1.0/(nu*var))*(x-mu)**2)**(-(nu+1)/2)
    return p

