from __future__ import division
# Copyright (c) 2017 Eero Siivola
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats,special
import scipy as sp
from GPy.likelihoods.link_functions import Log, Identity
from GPy.likelihoods.likelihood import Likelihood
from GPy.core.parameterization import Param
from GPy.core.parameterization.transformations import Logexp
from GPy.util.multioutput import index_to_slices

class CombinedLikelihood(Likelihood):
    """Combination of likelihoods, EP inference for all except gaussian likelihoods"""
    def __init__(self, likelihoods, name='Combined_likelihood'):
        super(Likelihood, self).__init__(name=name)
        self.likelihoods = likelihoods
        ind=[]
        for i in range(0,len(likelihoods)):
            unique = True
            for j in range(0,i):
                if likelihoods[i] is likelihoods[j]:
                    unique=False
            if unique is True:
                ind.append(i)  
        self.link_parameters(*[likelihoods[i] for i in ind])
    
    def moments_match_ep(self, data_i, tau_i, v_i, Y_metadata_i=None):
        return self.likelihoods[Y_metadata_i["likelihood"]].moments_match_ep(data_i, tau_i, v_i, Y_metadata_i)
    
    def get_fixed_gaussian(self, X, Y, index_dim=-1): #fixed_index, fixed_gaussian_v, fixed_gaussian_tau
        likelihood_i = np.array(X[:,index_dim], dtype=np.int)
        gaussian_i = [i for i in range(len(self.likelihoods)) if self.likelihoods[i].name is 'Gaussian_noise']
        index = np.asarray([i for i in range(X.size[0]) if X[i,index_dim] in gaussian_i], dtype=np.int)
        tau = np.asarray([1./self.likelihoods[i].variance for i in likelihood_i if i in gaussian_i], dtype=np.float64)
        v = np.asarray([Y[i] for i in likelihood_i if i in gaussian_i], dtype=np.float64) * tau
        
        

    def predictive_values(self, mu_list, var_list, full_cov=False, Y_metadata_list=None):
        nl = len(self.likelihoods)
        mu = [None]*nl
        var = [None]*nl
        Y_metadata_list = [None]*nl if Y_metadata_list is None else Y_metadata_list
        for i in range(0,nl):
            mu[i], var[i] = self.likelihoods[i].predictive_values(mu_list[i], var_list[i], full_cov, Y_metadata_list[i])
        return mu, var

    def predictive_mean(self, mu, sigma, Y_metadata):
        nl = len(self.likelihoods)
        mu = [None]*nl
        for i in range(0,nl):
            mu[i] = self.likelihoods[i].predictive_mean(mu_list[i], sigma_list[i])
        return mu, var

    def predictive_variance(self, mu, sigma, Y_metadata):
        nl = len(self.likelihoods)
        var = [None]*nl
        predictive_mean_list = [None]*nl if predictive_mean_list is None else predictive_mean_list 
        for i in range(0,nl):
            var[i] = self.likelihoods[i].predictive_variance(mu_list[i], sigma_list[i], predictive_mean_list[i])
        return var

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata):
        nl = len(self.likelihoods)
        quant = [None]*nl
        Y_metadata_list = [None]*nl if Y_metadata_list is None else Y_metadata_list
        for i in range(0,nl):
            quant[i] = self.likelihoods[i].predictive_quantiles(mu_list[i], var_list[i], quantiles, Y_metadata_list[i])
        return quant