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
    
    def get_fixed_gaussian(self, X_list): #fixed_index, fixed_gaussian_v, fixed_gaussian_tau
        nl = len(self.likelihoods)
        t1 = 0
        index = np.empty(0)
        tau = np.empty(0)
        v = np.empty(0)
        for i in range(0,nl):
            t2 = t1 if X_list[i] is None else X_list[i].size+t1 
            if self.likelihoods[i].name is 'Gaussian_noise':
                index = np.r_[index, range(t1, t2)]
                tau = np.r_[tau, np.ones(X_list[i].size, dtype=np.float64)*1./self.likelihoods[i].variance]
                v = np.r_[v, np.ones(X_list[i].size, dtype=np.float64)*1./self.likelihoods[i].variance]
            t1 = t2
        return index, tau, v
        
    def predictive_values(self, mu_list, var_list, full_cov=False, Y_metadata_list=None):
        nl = len(self.likelihoods)
        mu = [None]*nl
        var = [None]*nl
        Y_metadata_list = [None]*nl if Y_metadata_list is None else Y_metadata_list
        for i in range(0,nl):
            mu[i], var[i] = self.likelihoods[i].predictive_values(mu_list[i], var_list[i], full_cov, Y_metadata_list[i])
        return mu, var

    def predictive_mean(self, mu_list, sigma_list):
        nl = len(self.likelihoods)
        mu = [None]*nl
        for i in range(0,nl):
            mu[i] = self.likelihoods[i].predictive_mean(mu_list[i], sigma_list[i])
        return mu, var

    def predictive_variance(self, mu_list, sigma_list, predictive_mean_list=None):
        nl = len(self.likelihoods)
        var = [None]*nl
        predictive_mean_list = [None]*nl if predictive_mean_list is None else predictive_mean_list 
        for i in range(0,nl):
            var[i] = self.likelihoods[i].predictive_variance(mu_list[i], sigma_list[i], predictive_mean_list[i])
        return var

    def predictive_quantiles(self, mu_list, var_list, quantiles, Y_metadata_list=None):
        nl = len(self.likelihoods)
        quant = [None]*nl
        Y_metadata_list = [None]*nl if Y_metadata_list is None else Y_metadata_list
        for i in range(0,nl):
            quant[i] = self.likelihoods[i].predictive_quantiles(mu_list[i], var_list[i], quantiles, Y_metadata_list[i])
        return quant