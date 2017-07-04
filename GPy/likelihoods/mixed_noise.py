# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats, special
from . import link_functions
from .likelihood import Likelihood
from .gaussian import Gaussian
from ..core.parameterization import Param
from paramz.transformations import Logexp
from ..core.parameterization import Parameterized
import itertools

class MixedNoise(Likelihood):
    def __init__(self, likelihoods_list, name='mixed_noise'):
        super(Likelihood, self).__init__(name=name)
        groups = []
        for i in range(len(likelihoods_list)):
            unique = True
            for j in range(i):
                if likelihoods_list[i] is likelihoods_list[j]:
                    unique=False
                    for li in groups:
                        if j in li:
                            li.append(i)
                            break
            if unique is True:
                groups.append([i])
        self.gp_link = None
        self.groups=groups
        self.link_parameters(*[likelihoods_list[g[0]] for g in groups])
        self.likelihoods_list = likelihoods_list
        self.log_concave = False

    def moments_match_ep(self, data_i, tau_i, v_i, Y_metadata_i):
        return self.likelihoods_list[Y_metadata_i["output_index"][0]].moments_match_ep(data_i, tau_i, v_i, Y_metadata_i)
    
    def get_fixed_gaussian(self, X, Y, index_dim=-1): #fixed_index, fixed_gaussian_v, fixed_gaussian_tau
        likelihood_i = np.array(X[:,index_dim], dtype=np.int)
        gaussian_i = [i for i in range(len(self.likelihoods_list)) if self.likelihoods_list[i].name is 'Gaussian_noise']
        index = np.asarray([i for i in range(X.shape[0]) if X[i,index_dim] in gaussian_i], dtype=np.int)
        tau = np.asarray([1./self.likelihoods_list[int(X[i, index_dim])].variance for i in index], dtype=np.float64)
        v = np.array([Y[i] for i in index]) * tau
        return index, tau, v
        
    def gaussian_variance(self, Y_metadata):
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index'].flatten()
        variance = np.zeros(ind.size)
        for lik, j in zip(self.likelihoods_list, range(len(self.likelihoods_list))):
            variance[ind==j] = lik.variance
        return variance

    def betaY(self,Y,Y_metadata):
        #TODO not here.
        return Y/self.gaussian_variance(Y_metadata=Y_metadata)[:,None]
    
    def reset_gradients(self):
        for likelihood in self.likelihoods_list:
            likelihood.reset_gradients()

    def ep_gradients(self, Y, cav_tau, cav_v, dL_dKdiag, Y_metadata=None, quad_mode='gk', boost_grad=1.):
        ind = Y_metadata['output_index'].flatten()
        grads = np.zeros((self.size) )
        j=0
        for i in range(len(self.groups)):
            s = j + self.likelihoods_list[self.groups[i][0]].size
            if s > j:
                for k in self.groups[i]:
                    temp = self.likelihoods_list[k].ep_gradients(Y[ind==k,:], cav_tau[ind==k], cav_v[ind==k], dL_dKdiag = dL_dKdiag[ind==k], Y_metadata=Y_metadata, quad_mode=quad_mode, boost_grad=boost_grad)
                    grads[j:s] += self.likelihoods_list[k].ep_gradients(Y[ind==k,:], cav_tau[ind==k], cav_v[ind==k], dL_dKdiag = dL_dKdiag[ind==k], Y_metadata=Y_metadata, quad_mode=quad_mode, boost_grad=boost_grad)
            j=s
        return grads   

    def update_gradients(self, gradients, reset=True):
        if reset:
            self.reset_gradients()
        j=0
        for i in range(len(self.groups)):
            s = j + self.likelihoods_list[self.groups[i][0]].size
            self.likelihoods_list[self.groups[i][0]].update_gradients( gradients[j:s], False)
            j = s

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata):
       #assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
       ind = Y_metadata['output_index'].flatten()
       grad = np.array([self.likelihoods_list[i].exact_inference_gradients(dL_dKdiag[ind==i], None) for i in range(len(self.likelihoods_list))])
       return self._collide_to_groups(grad)

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        mu_new = np.zeros(mu.shape )
        var_new = np.zeros(var.shape )
        for j in outputs:
            m, v = self.likelihoods_list[j].predictive_values(mu[ind==j,:], var[ind==j,:], full_cov, Y_metadata=None)
            mu_new[ind==j,:] = m
            var_new[ind==j,:] = v
        return mu_new, var_new
    
        #for i in range(len(ind)):
            #mu[i,:], var[i,:] = self.likelihoods_list[ind[i]].predictive_values(mu[i,:], var[i,:], full_cov)
        #return mu, var
        #_variance = np.array([self.likelihoods_list[j].variance for j in ind ])
        #if full_cov:
            #var += np.eye(var.shape[0])*_variance
        #else:
            #var += _variance
        #return mu, var

    def predictive_variance(self, mu, sigma, Y_metadata):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        var = np.zeros( (sigma.size) )
        for j in outputs:
            v = self.likelihoods_list[j].predictive_variance(mu[ind==j,:],
                sigma[ind==j,:],Y_metadata=None)
            var[ind==j,:] = np.hstack(v)
        return [v[:,None] for v in var.T]
    
        #ind = Y_metadata['output_index'].flatten()
        #for i in range(len(ind)):
            #var[i,:] = self.likelihoods_list[ind[i]].predictive_variance(mu[i,:], sigma[i,:])
        #return var
        #_variance = self.gaussian_variance(Y_metadata)
        #return _variance + sigma**2

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        Q = np.zeros( (mu.size,len(quantiles)) )
        for j in outputs:
            q = self.likelihoods_list[j].predictive_quantiles(mu[ind==j,:],
                var[ind==j,:],quantiles,Y_metadata=None)
            Q[ind==j,:] = np.hstack(q)
        return [q[:,None] for q in Q.T]

    def samples(self, gp, Y_metadata):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        N1, N2 = gp.shape
        Ysim = np.zeros((N1,N2))
        ind = Y_metadata['output_index'].flatten()
        for j in np.unique(ind):
            flt = ind==j
            gp_filtered = gp[flt,:]
            n1 = gp_filtered.shape[0]
            lik = self.likelihoods_list[j]
            _ysim = np.array([np.random.normal(lik.gp_link.transf(gpj), scale=np.sqrt(lik.variance), size=1) for gpj in gp_filtered.flatten()])
            Ysim[flt,:] = _ysim.reshape(n1,N2)
        return Ysim

    def logpdf(self, f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        if ind.shape[0]==1:
            ind = ind[0]*np.ones(f.shape[0])
            y = y*np.ones(f.shape)
        lpdf = np.zeros( (f.size))
        for j in outputs:
            lpdf[ind==j] = self.likelihoods_list[j].logpdf(f[ind==j,:], y[ind==j,:], Y_metadata=None)
        return lpdf

    def dlogpdf_dtheta(self, f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        if ind.shape[0]==1:
            ind = ind[0]*np.ones(f.shape[0])
            y = y*np.ones(f.shape)
        pdf = np.zeros((self.size, f.shape[0], f.shape[1]) )
        j=0
        for i in range(len(self.groups)):
            s = j + self.likelihoods_list[self.groups[i][0]].size
            if s > j:
                for k in self.groups[i]:
                    pdf[j:s,ind == k,:] = self.likelihoods_list[k].dlogpdf_dtheta(f[ind==k,:], y[ind==j,:], Y_metadata=None)
            j=s
        return pdf
    
    def site_derivatives_ep(self,obs,tau,v,Y_metadata_i=None, gh_points=None, dL_dKdiag = None):
        # "calculate site derivatives E_f{dp(y_i|f_i)/dr} where r is a parameter of the likelihood term."
        # "writing it explicitly "
        # use them for gaussian-hermite quadrature, or gaussian-kronrod quadrature !!!
        mu = v/tau
        sigma2 = 1./tau

        if gh_points is None:
            gh_x, gh_w = self._gh_points(32)
        else:
            gh_x, gh_w = gh_points

        # X = gh_x[None,:]*np.sqrt(2.*v[:,None]) + m[:,None]
        X = gh_x[:,None] #Nx1

        logp = self.logpdf(X, obs, Y_metadata=Y_metadata_i)
        dlogp_dtheta = self.dlogpdf_dtheta(X, obs, Y_metadata=Y_metadata_i)

        F = np.dot(logp, gh_w)/np.sqrt(np.pi)
        dF_dtheta_i = np.dot(dlogp_dtheta.reshape((self.size,-1)), gh_w)/np.sqrt(np.pi)
        ind = Y_metadata_i['output_index'].flatten()
        if self.likelihoods_list[ind[0]].name is 'Gaussian_noise':
            dF_dtheta_i[0] = dL_dKdiag
        return dF_dtheta_i

    def _collide_to_groups(self, orig):
        new = []
        for group in self.groups:
            temp = None
            for ind in group:
                temp = temp + orig[ind] if (temp is not None and orig[ind] is not None) else (orig[ind] if orig[ind] is not None else None)
            if temp is not None:
                new.append(temp)
        return np.array(new)