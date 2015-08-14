import numpy as np
from studentpdf import studentpdf

class gaussian1D():
    def __init__(self):
        self.num_model_params = 0
        self.post_params  = np.asarray([])
        self.dpost_params = np.asarray([])
        self.predprobs    = np.asarray([])
        self.dpredprobs   = np.asarray([])
        self.dmessage     = []

    def init_f(self,T,D,theta_m):
        self.theta            = np.asarray(theta_m)
        self.post_params      = np.asarray(theta_m)
        self.num_model_params = len(theta_m)
        self.dmessage         = np.eye(4)

    def update(self,xt,needDerivatives=False):
        mus    = self.post_params[0] # N x 1. [x]
        kappas = self.post_params[1] # N x 1. [points]\
        alphas = self.post_params[2] # N x 1. [points]
        betas  = self.post_params[3] # N x 1. [x^2]

        # Posterior update rules found in p. 29 of
        # http://www.stat.columbia.edu/~cook/movabletype/mlm/CONJINTRnew#2BTEX.pdf
        # except their betas are scale and we use inverse scale. Set n = 1 to attain the online update rules. Note that SS = 0 when n = 1.
        # TODO rename kappas taus for consistency with document.

        # p(mean X|X) = N(mus, kappas), mus = mean, kappas = precision on E[X]
        # Bayesian update on mus when n = 1. N + 1 x 1. [x]
        mus_new    = [self.theta[0]].extend((kappas*mus + xt)/(kappas + 1))
        kappas_new = [self.theta[1], kappas+1] # N + 1 x 1. [points]
        # p(precision X|X) = Gamma(alphas, betas), alphas = shape, betas = inv scale
        # => E[precision X] = alpha / beta => E[var X] = beta / alpha.
        alphas_new = [self.theta[2], alphas + 0.5] # N + 1 x 1. [points]
        # N + 1 x 1. [x^2]
        betas_new  = [self.theta[3], betas + (kappas*(xt - mus)**2)/(2.0*(kappas+1))]
        print "I AM HERE!!!!"
        self.post_params = [mus_new, kappas_new, alphas_new, betas_new] # N + 1 x 4. [mixed]

        if needDerivatives:
          dmu_dmu0       = np.squeeze(self.dpost_params[0, 0, :]) # N x 1
          dmu_dkappa0    = np.squeeze(self.dpost_params[0, 1, :]) # N x 1
          dkappa_dkappa0 = np.squeeze(self.dpost_params[1, 1, :]) # N x 1
          dbeta_dmu0     = np.squeeze(self.dpost_params[3, 0, :]) # N x 1
          dbeta_dkappa0  = np.squeeze(self.dpost_params[3, 1, :]) # N x 1

          # Update the components which need updating
          dmu_dmu0_new = (kappas/(kappas + 1))*dmu_dmu0 # d_mu / d_mu0. N x 1
          dmu_dkappa0_new = (dkappa_dkappa0*mus + dmu_dkappa0*kappas) / (kappas + 1) - ((kappas * mus + xt)*dkappa_dkappa0) / ((kappas + 1)**2) # N x 1
          dbeta_dmu0_new = dbeta_dmu0 - ((kappas*(xt - mus))/(kappas + 1)*dmu_dmu0) # N x 1

          den          = 2.0*(kappas + 1) # N x 1
          num          = kappas*(xt - mus)**2 # N x 1
          dden_dkappa0 = 2.0*dkappa_dkappa0 # N x 1
          dnum_dkappa0 = dkappa_dkappa0*(xt - mus)**2 + 2.0*kappas*(mus - xt)*dmu_dkappa0 # N x 1
          QR           = (dnum_dkappa0*den - dden_dkappa0*num)/ den**2 # N x 1
          dbeta_dkappa0_new = dbeta_dkappa0 + QR # N x 1

          self.dpost_params[0,0,:] = dmu_dmu0_new # 1 x 1 x N
          self.dpost_params[0,1,:] = dmu_dkappa0_new # 1 x 1 x N
          self.dpost_params[3,0,:] = dbeta_dmu0_new # 1 x 1 x N
          self.dpost_params[3,1,:] = dbeta_dkappa0_new # 1 x 1 x N

          self.dpost_params[:,:,:self.dpost_params.shape[2]+1] = self.dpost_params  # 1 x 1 x N
          self.dpost_params[:,:,0] = np.eye(self.dpost_params.shape[0]) # 4 x 4 x 1

    def predict(self,X,needDerivatives=False):
      N = self.post_params.shape[0] # 1 x 1

      mus    = self.post_params[0] # N x 1. [x]
      kappas = self.post_params[1] # N x 1. [points]
      alphas = self.post_params[2] # N x 1. [points]
      betas  = self.post_params[3] # N x 1. [x^2]

      # TODO verify this is correct by citing reference with posterior predictive
      # However, probably correct since we get the same lml under random
      # permutations of the data => coherence.
      # N x 1. [x^2]
      predictive_variance = betas*(kappas+1)/(alphas*kappas)
      df                  = 2.0*alphas # N x 1. [points]

      if not needDerivatives:
        self.predprobs = studentpdf(xnew, mus, predictive_variance, df) # N x 1. [P/x]
      else:
        self.predprobs, dtpdf = studentpdf(X, mus, predictive_variance, df, nargout=2) # N x 1. [P/x]
        dmu_dtheta    = np.transpose(self.dpost_params[0,:,:], axes=[2,1,0]) # N x 4
        dkappa_dtheta = np.transpose(self.dpost_params[1,:,:], axes=[2,1,0]) # N x 4
        dalpha_dtheta = np.transpose(self.dpost_params[2,:,:], axes=[2,1,0]) # N x 4
        dbeta_dtheta  = np.transpose(self.dpost_params[3,:,:], axes=[2,1,0]) # N x 4

        dnu_dtheta = 2.0 * dalpha_dtheta # N x 4

        # TODO use rmult and eliminate the for loop
        dpv_dtheta = np.zeros((N, 4))
        for ii in range(4):
          QRpart = (dbeta_dtheta[:,ii]*alphas - betas*dalpha_dtheta[:,ii])/ alphas**2 # N x 1
          dpv_dtheta[:,ii] = -(betas/(alphas*kappas**2))*dkappa_dtheta[:,ii] + (1 + 1/kappas)*QRpart # N x 1

        # TODO use rmult and eliminate the for loop
        dp_dtheta = np.zeros((N, 4))
        for ii in range(4):
          # dp/dtheta_i = dp/dmu * dmu/dtheta_i + dp/dsigma2 * dsigma2/dtheta_i +
          # dp/dnu + dnu/dtheta_i
          # N x 1
          dp_dtheta[:,ii] = dtpdf[:,0]*dmu_dtheta[:,ii] + dtpdf[:, 1]*dpv_dtheta[:,ii] + dtpdf[:, 2]*dnu_dtheta[:,ii]
        self.dpredprobs = dp_dtheta

