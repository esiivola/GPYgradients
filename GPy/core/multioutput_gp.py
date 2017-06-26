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
    def __init__(self, X_list, Y_list, kernel_list, likelihood_list, name='multioutputgp', kernel_cross_covariances={}):
        #Input and Output
        X,Y,self.output_index = util.multioutput.build_XY(X_list,Y_list)
        Ny = len(Y_list)
        
        assert isinstance(kernel_list, list)
        kernel = kern.MultioutputKern(kernels=kernel_list, cross_covariances=kernel_cross_covariances)

        assert isinstance(likelihood_list, list)
        likelihood = likelihoods.MixedNoise(likelihood_list)
        
        super(MultioutputGP, self).__init__(X,Y,kernel,likelihood, Y_metadata={'output_index':self.output_index}, inference_method = expectation_propagation.MultioutputEP())# expectation_propagation.MultioutputEP()) # expectation_propagation.EP())                             
                                            #expectation_propagation.MultioutputEP())

    def predict_noiseless(self,  Xnew, full_cov=False, Y_metadata=None, kern=None):
        X, _, _  = util.multioutput.build_XY(Xnew,None)
        return super(MultioutputGP, self).predict_noiseless(X, full_cov, Y_metadata, kern)