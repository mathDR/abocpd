from numpy import ones_like

def constant_hazard(r, lam):
  return 1.0/lam * ones_like(r)

