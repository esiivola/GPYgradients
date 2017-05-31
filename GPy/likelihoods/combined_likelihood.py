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
    
    def predictive_values(self, mu_list, var_list, full_cov=False, Y_metadata_list=None):
        mu = []
        return mu, var

    def predictive_mean(self, mu_list, sigma_list):
        return mu

    def predictive_variance(self, mu_list, sigma_list, predictive_mean_list=None):
        return self.variance + sigma**2

    def predictive_quantiles(self, mu_list, var_list, quantiles, Y_metadata_list=None):
        return  [stats.norm.ppf(q/100.)*np.sqrt(var + self.variance) + mu for q in quantiles]    