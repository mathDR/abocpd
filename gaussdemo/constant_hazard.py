from numpy import ones_like

def constant_hazard(r, lambda):
  return 1.0/lambda * ones_like(r)

