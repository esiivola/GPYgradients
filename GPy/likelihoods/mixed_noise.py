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

    def update_gradients(self, gradients):
        self.gradient = gradients

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata):
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index'].flatten()
        grad = np.array([dL_dKdiag[ind==i].sum() for i in range(len(self.likelihoods_list))])
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
            v = self.likelihoods_list[j].predictive_varaince(mu[ind==j,:],
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

    def _collide_to_groups(self, orig):
        new = []
        for group in self.groups:
            temp = 0
            for ind in group:
                temp += orig[ind]
            new.append(temp)
        return np.array(new)