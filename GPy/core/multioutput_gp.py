# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .model import Model
from .parameterization.variational import VariationalPosterior
from .mapping import Mapping
from .. import likelihoods
from .. import kern
from ..inference.latent_function_inference import exact_gaussian_inference, expectation_propagation
from ..util.normalizer import Standardize
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
    def __init__(self, X_list, Y_list, kernel_list, likelihood_list, name='multioutputgp', Y_metadata_list=None, kernel_cross_covariances={}):
        super(GP, self).__init__(name)
        nl = len(X_list)
        X = [None]*nl
        Y = [None]*nl
        for i in range(0,nl): 
            X[i] = ObsAr(X_list[i])
            Y[i] = ObsAr(Y_list[i])
        self.X = X
        self.Y = Y
        self.Y_normalized = Y
        #self.num_data, self.input_dim = self.X.shape
        
        self.normalizer = None
        self.mean_function = None

        #_, self.output_dim = self.Y.shape

        assert ((Y_metadata_list is None) or isinstance(Y_metadata_list, list))
        self.Y_metadata = Y_metadata_list

        assert isinstance(kernel_list, list)
        self.kern = kern.MultioutputKern(kern_list=kernel_list, cross_covariances=kernel_cross_covariances)

        assert isinstance(likelihood_list, list)
        self.likelihood = likelihoods.CombinedLikelihood(likelihood_list)

        self.inference_method = expectation_propagation.MultioutputEP(self.kern, self.X, self.likelihood, self.Y)

        logger.info("adding kernel and likelihood as parameters")
        self.link_parameter(self.kern)
        self.link_parameter(self.likelihood)
        self.posterior = None