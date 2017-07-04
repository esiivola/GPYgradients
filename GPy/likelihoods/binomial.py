# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..util.univariate_Gaussian import std_norm_pdf, std_norm_cdf
from . import link_functions
from .likelihood import Likelihood
from scipy import special

class Binomial(Likelihood):
    """
    Binomial likelihood

    .. math::
        p(y_{i}|\\lambda(f_{i})) = \\lambda(f_{i})^{y_{i}}(1-f_{i})^{1-y_{i}}

    .. Note::
        Y takes values in either {-1, 1} or {0, 1}.
        link function should have the domain [0, 1], e.g. probit (default) or Heaviside

    .. See also::
        likelihood.py, for the parent class
    """
    def __init__(self, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Probit()

        super(Binomial, self).__init__(gp_link, 'Binomial')

    def pdf_link(self, inv_link_f, y, Y_metadata):
        """
        Likelihood function given inverse link of f.

        .. math::
            py(_{i}|\\lambda(f_{i})) = \\lambda(f_{i})^{y_{i}}(1-f_{i})^{1-y_{i}}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata must contain 'trials'
        :returns: likelihood evaluated for this point
        :rtype: float

        .. Note:
            Each y_i must be in {0, 1}
        """
        return np.exp(self.logpdf_link(inv_link_f, y, Y_metadata))

    def logpdf_link(self, inv_link_f, y, Y_metadata=None):
        """
        Log Likelihood function given inverse link of f.

        .. math::
            \\ln p(y_{i}|\\lambda(f_{i})) = y_{i}\\log\\lambda(f_{i}) + (1-y_{i})\\log (1-f_{i})

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata must contain 'trials'
        :returns: log likelihood evaluated at points inverse link of f.
        :rtype: float
        """
        N = np.ones(y.shape) if Y_metadata is None else Y_metadata.get('trials', np.ones(y.shape))
        np.testing.assert_array_equal(N.shape, y.shape)

        nchoosey = special.gammaln(N+1) - special.gammaln(y+1) - special.gammaln(N-y+1)
        return nchoosey + y*np.log(inv_link_f) + (N-y)*np.log(1.-inv_link_f)

        #nchoosey = special.gammaln(N+1) - special.gammaln(y+1) - special.gammaln(N-y+1)
        #si = inv_link
        #if (inv_link_f < 1e-8 and  y>0) or (inv_link_f > 1-1e-8 and (N-y) >0):
            #return -np.inf
        #elif inv_link_f < 1e-8 or inv_link_f > 1-1e-8:
            #return nchoosey
        #else:
            #return nchoosey + y*np.log(inv_link_f) + (N-y)*np.log(1.-inv_link_f)

    def dlogpdf_dlink(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the pdf at y, given inverse link of f w.r.t inverse link of f.

        .. math::
            \\frac{d^{2}\\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)^{2}} = \\frac{y_{i}}{\\lambda(f)} - \\frac{(N-y_{i})}{(1-\\lambda(f))}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata must contain 'trials'
        :returns: gradient of log likelihood evaluated at points inverse link of f.
        :rtype: Nx1 array
        """
        N = np.ones(y.shape) if Y_metadata is None else Y_metadata.get('trials', np.ones(y.shape))
        np.testing.assert_array_equal(N.shape, y.shape)

        return y/inv_link_f - (N-y)/(1.-inv_link_f)

    def d2logpdf_dlink2(self, inv_link_f, y, Y_metadata=None):
        """
        Hessian at y, given inv_link_f, w.r.t inv_link_f the hessian will be 0 unless i == j
        i.e. second derivative logpdf at y given inverse link of f_i and inverse link of f_j  w.r.t inverse link of f_i and inverse link of f_j.


        .. math::
            \\frac{d^{2}\\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)^{2}} = \\frac{-y_{i}}{\\lambda(f)^{2}} - \\frac{(N-y_{i})}{(1-\\lambda(f))^{2}}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in binomial
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points inverse link of f.
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on inverse link of f_i not on inverse link of f_(j!=i)
        """
        N = np.ones(y.shape) if Y_metadata is None else Y_metadata.get('trials', np.ones(y.shape))
        np.testing.assert_array_equal(N.shape, y.shape)
        return -y/np.square(inv_link_f) - (N-y)/np.square(1.-inv_link_f)

    def d3logpdf_dlink3(self, inv_link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given inverse link of f w.r.t inverse link of f

        .. math::
            \\frac{d^{2}\\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)^{2}} = \\frac{2y_{i}}{\\lambda(f)^{3}} - \\frac{2(N-y_{i})}{(1-\\lambda(f))^{3}}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in binomial
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points inverse link of f.
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on inverse link of f_i not on inverse link of f_(j!=i)
        """
        N = np.ones(y.shape) if Y_metadata is None else Y_metadata.get('trials', np.ones(y.shape))
        np.testing.assert_array_equal(N.shape, y.shape)

        inv_link_f2 = np.square(inv_link_f)
        return 2*y/inv_link_f**3 - 2*(N-y)/(1.-inv_link_f)**3

    def dlogpdf_dtheta(self, f, y, Y_metadata=None):
        if isinstance(self.gp_link, link_functions.Probit):
            N = np.ones(y.shape) if Y_metadata is None else Y_metadata.get('trials', np.ones(y.shape))
            inv_link_f = self.gp_link.transf(f)
            return (y/inv_link_f - (N-y)/(1.-inv_link_f))*self.gp_link.dtransf_dtheta(f)
        else:
            return super(Binomial, self).dlogpdf_dtheta(f, y, Y_metadata)
    
    def update_gradients(self, partial, reset=True):
        if reset:
            self.reset_gradients()
        if isinstance(self.gp_link, link_functions.Probit):
            self.gp_link.update_gradients(partial, reset)
        else:
            raise NotImplementedError('This function call should not happen')

    def reset_gradients(self):
        self.gp_link.reset_gradients()

    def samples(self, gp, Y_metadata=None, **kw):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        N = Y_metadata['trials']
        Ysim = np.random.binomial(N, self.gp_link.transf(gp))
        return Ysim.reshape(orig_shape)
  
    def ep_gradients(self, Y, tau, v, Y_metadata=None, gh_points=None, boost_grad=1., dL_dKdiag=None):
        if isinstance(self.gp_link, link_functions.Probit):
            nu = self.gp_link.nu
            mu = v/tau
            sigma2 = 1./tau
            a = np.sqrt(1 + sigma2*(nu**2))
            z = mu/a
            return np.sum((self.gp_link.dtransf_df(z)/self.gp_link.transf(z))*mu/(nu*(sigma2*(nu**2)+1)**(3./2.)) )
        else:
            return super(Binomial, self).ep_gradients(Y, tau, v, Y_metadata=Y_metadata, gh_points=gh_points, boost_grad=boost_grad, dL_dKdiag=dL_dKdiag)

    def ep_gradients2(self, Y, tau, v, Y_metadata=None, gh_points=None, boost_grad=1., dL_dKdiag=None):
        if isinstance(self.gp_link, link_functions.Probit):
            nu = self.gp_link.nu
            mu = v/tau
            sigma2 = 1./tau
            a = np.sqrt(1 + sigma2/(nu**2))
            z = mu/a
            return -1.0*np.sum((1.0/self.gp_link.transf(z))*self.gp_link.dtransf_df(z)*nu*z/(sigma2+nu**2))
        else:
            return super(Binomial, self).ep_gradients(Y, tau, v, Y_metadata=Y_metadata, gh_points=gh_points, boost_grad=boost_grad, dL_dKdiag=dL_dKdiag)
  
    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        if isinstance(self.gp_link, link_functions.Probit):

            if gh_points is None:
                gh_x, gh_w = self._gh_points()
            else:
                gh_x, gh_w = gh_points


            gh_w = gh_w / np.sqrt(np.pi)
            shape = m.shape
            C = np.atleast_1d(Y_metadata['trials'])
            m,v,Y, C = m.flatten(), v.flatten(), Y.flatten()[:,None], C.flatten()[:,None]
            X = gh_x[None,:]*np.sqrt(2.*v[:,None]) + m[:,None]
            p = std_norm_cdf(X)
            p = np.clip(p, 1e-9, 1.-1e-9) # for numerical stability
            N = std_norm_pdf(X)
            #TODO: missing nchoosek coefficient! use gammaln?
            F = (Y*np.log(p) + (C-Y)*np.log(1.-p)).dot(gh_w)
            NoverP = N/p
            NoverP_ = N/(1.-p)
            dF_dm = (Y*NoverP - (C-Y)*NoverP_).dot(gh_w)
            dF_dv = -0.5* ( Y*(NoverP**2 + NoverP*X) + (C-Y)*(NoverP_**2 - NoverP_*X) ).dot(gh_w)
            return F.reshape(*shape), dF_dm.reshape(*shape), dF_dv.reshape(*shape), None
        else:
            raise NotImplementedError

    def moments_match_ep(self,obs,tau,v,Y_metadata_i=None):
        """
        Calculation of moments using quadrature

        :param obs: observed output
        :param tau: cavity distribution 1st natural parameter (precision)
        :param v: cavity distribution 2nd natural paramenter (mu*precision)
        """
        #Compute first integral for zeroth moment.
        #NOTE constant np.sqrt(2*pi/tau) added at the end of the function
        if isinstance(self.gp_link, link_functions.Probit) and (Y_metadata_i is None or int(Y_metadata_i.get('trials', 1)) == int(1)): #Special case for probit likelihood. Can be found from Riihimaki et Vehtari 2010
            nu = self.gp_link.nu
            mu = v/tau
            sigma2 = 1./tau
            a = np.sqrt(1 + sigma2*(nu**2))
            z = obs*mu/a
            normc_z = self.gp_link.transf(z)
            m0 = normc_z
            normp_z = self.gp_link.dtransf_df(z)
            m1 = mu + (obs*sigma2*normp_z)/(normc_z*a)
            #print('tau: {}, v: {}, nu: {}, z: {}, normc_z: {}, normp_z: {}'.format(tau, v, nu.values, z, normc_z, normp_z))
            m2 = sigma2 - ((sigma2**2)*normp_z)/((1./(nu**2)+sigma2)*normc_z)*(z + normp_z/(nu**2)/normc_z)
            #print("m0: {}, m1: {}, m2: {}".format(m0,m1,m2))
            #m0a, m1a, m2a =  super(Binomial, self).moments_match_ep(obs,tau,v,Y_metadata_i)
            #print("m0a: {}, m1a: {}, m2a: {}".format(m0a,m1a,m2a))
            return m0, m1, m2
        else:
            return super(Binomial, self).moments_match_ep(obs,tau,v,Y_metadata_i)

    def moments_match_ep2(self,obs,tau,v,Y_metadata_i=None):
        """
        Calculation of moments using quadrature

        :param obs: observed output
        :param tau: cavity distribution 1st natural parameter (precision)
        :param v: cavity distribution 2nd natural paramenter (mu*precision)
        """
        #Compute first integral for zeroth moment.
        #NOTE constant np.sqrt(2*pi/tau) added at the end of the function
        if isinstance(self.gp_link, link_functions.Probit) and (Y_metadata_i is None or int(Y_metadata_i.get('trials', 1)) == int(1)): #Special case for probit likelihood. Can be found from Riihimaki et Vehtari 2010
            nu = self.gp_link.nu
            mu = v/tau
            sigma2 = 1./tau
            a = np.sqrt(1 + sigma2/(nu**2))
            z = obs*mu/a
            normc_z = self.gp_link.transf(z)
            m0 = normc_z
            normp_z = self.gp_link.dtransf_df(z)
            m1 = mu + (obs*sigma2*normp_z)/(normc_z*a)   
            m2 = sigma2 - ((sigma2**2)*normp_z)/((nu**2+sigma2)*normc_z)*(z + normp_z*nu**2/normc_z)
            #print("m0: {}, m1: {}, m2: {}".format(m0,m1,m2))
            #m0a, m1a, m2a =  super(Binomial, self).moments_match_ep(obs,tau,v,Y_metadata_i)
            #print("m0a: {}, m1a: {}, m2a: {}".format(m0a,m1a,m2a))
            return m0, m1, m2
        else:
            return super(Binomial, self).moments_match_ep(obs,tau,v,Y_metadata_i)