# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
from .model import Model
from .parameterization.variational import VariationalPosterior
from .mapping import Mapping
from .. import likelihoods
from .. import kern
from ..inference.latent_function_inference import exact_gaussian_inference, expectation_propagation
from ..util.normalizer import Standardize
from .. import util
from paramz import ObsAr
from .gp import GP

import logging
import warnings
logger = logging.getLogger("GP")

class MultioutputGP(GP):
    """
    General purpose Gaussian process model

    :param X: input observations
    :param Y: output observations
    :param kernel: a GPy kernel, defaults to rbf+white
    :param likelihood: a GPy likelihood
    :param inference_method: The :class:`~GPy.inference.latent_function_inference.LatentFunctionInference` inference method to use for this GP
    :rtype: model object
    :param Norm normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is None, we will normalize using Standardize.
        If normalizer is False, no normalization will be done.

    .. Note:: Multiple independent outputs are allowed using columns of Y


    """
    def __init__(self, X_list, Y_list, kernel_list, likelihood_list, name='multioutputgp', kernel_cross_covariances={}, inference_method=None):
        #Input and Output
        X,Y,self.output_index = util.multioutput.build_XY(X_list,Y_list)
        Ny = len(Y_list)
        
        assert isinstance(kernel_list, list)
        kernel = kern.MultioutputKern(kernels=kernel_list, cross_covariances=kernel_cross_covariances)

        assert isinstance(likelihood_list, list)
        likelihood = likelihoods.MixedNoise(likelihood_list)
        
        if inference_method is None:
            inference_method = expectation_propagation.EP() 
        
        super(MultioutputGP, self).__init__(X,Y,kernel,likelihood, Y_metadata={'output_index':self.output_index}, inference_method = inference_method)# expectation_propagation.MultioutputEP()) # expectation_propagation.EP())                             
                                            #expectation_propagation.MultioutputEP())

    def predict_noiseless(self,  Xnew, full_cov=False, Y_metadata=None, kern=None):
        if isinstance(Xnew, list):
            Xnew, _, ind  = util.multioutput.build_XY(Xnew,None)
            if Y_metadata is None:
                Y_metadata={'output_index': ind}
        return super(MultioutputGP, self).predict_noiseless(Xnew, full_cov, Y_metadata, kern)
    
    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True):
        if isinstance(Xnew, list):
            Xnew, _, ind  = util.multioutput.build_XY(Xnew,None)
            if Y_metadata is None:
                Y_metadata={'output_index': ind}
        return super(MultioutputGP, self).predict(Xnew, full_cov, Y_metadata, kern, likelihood, include_likelihood)
    
    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None, kern=None, likelihood=None):
        if isinstance(X, list):
            X, _, ind  = util.multioutput.build_XY(X,None)
            if Y_metadata is None:
                Y_metadata={'output_index': ind}
        return super(MultioutputGP, self).predict_quantiles(X, quantiles, Y_metadata, kern, likelihood)
    
    def predictive_gradients(self, Xnew, kern=None):
        if isinstance(Xnew, list):
            Xnew, _, ind  = util.multioutput.build_XY(Xnew, None)
            if Y_metadata is None:
                Y_metadata={'output_index': ind}
        return super(MultioutputGP, self).predictive_gradients(Xnew, kern)
    
    
    def set_XY(self, X=None, Y=None):
        if isinstance(X, list):
            X, _, self.output_index  = util.multioutput.build_XY(X, None)
        if isinstance(Y, list):
            _, Y, self.output_index  = util.multioutput.build_XY(Y, Y)      
                
        self.update_model(False)
        if Y is not None:
            self.Y = ObsAr(Y)
            self.Y_normalized = self.Y
        if X is not None:
            self.X = ObsAr(X)
            
        self.Y_metadata={'output_index': self.output_index}
                
        self.inference_method.reset()
        self.update_model(True)