# -*- coding: utf-8 -*-
# Copyright (c) 2015, Alex Grigorevskiy
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
The standard periodic kernel which mentioned in:

[1] Gaussian Processes for Machine Learning, C. E. Rasmussen, C. K. I. Williams.
The MIT Press, 2005.


[2] Introduction to Gaussian processes. D. J. C. MacKay. In C. M. Bishop, editor,
Neural Networks and Machine Learning, pages 133-165. Springer, 1998.
"""

from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp

import numpy as np

class StdPeriodic(Kern):
    """
    Standart periodic kernel

    .. math::

       k(x,y) = \theta_1 \exp \left[  - \frac{1}{2} \sum_{i=1}^{input\_dim}
       \left( \frac{\sin(\frac{\pi}{T_i} (x_i - y_i) )}{l_i} \right)^2 \right] }

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance :math:`\theta_1` in the formula above
    :type variance: float
    :param period: the vector of periods :math:`\T_i`. If None then 1.0 is assumed.
    :type period: array or list of the appropriate size (or float if there is only one period parameter)
    :param lengthscale: the vector of lengthscale :math:`\l_i`. If None then 1.0 is assumed.
    :type lengthscale: array or list of the appropriate size (or float if there is only one lengthscale parameter)
    :param ARD1: Auto Relevance Determination with respect to period.
        If equal to "False" one single period parameter :math:`\T_i` for
        each dimension is assumed, otherwise there is one lengthscale
        parameter per dimension.
    :type ARD1: Boolean
    :param ARD2: Auto Relevance Determination with respect to lengthscale.
        If equal to "False" one single lengthscale parameter :math:`l_i` for
        each dimension is assumed, otherwise there is one lengthscale
        parameter per dimension.
    :type ARD2: Boolean
    :param active_dims: indices of dimensions which are used in the computation of the kernel
    :type active_dims: array or list of the appropriate size
    :param name: Name of the kernel for output
    :type String
    :param useGPU: whether of not use GPU
    :type Boolean
    """

    def __init__(self, input_dim, variance=1., period=None, lengthscale=None, ARD1=False, ARD2=False, active_dims=None, name='std_periodic',useGPU=False):
        super(StdPeriodic, self).__init__(input_dim, active_dims, name, useGPU=useGPU)
        self.input_dim = input_dim
        self.ARD1 = ARD1 # correspond to periods
        self.ARD2 = ARD2 # correspond to lengthscales

        self.name = name

        if self.ARD1 == False:
            if period is not None:
                period = np.asarray(period)
                assert period.size == 1, "Only one period needed for non-ARD kernel"
            else:
                period = np.ones(1.0)
        else:
            if period is not None:
                period = np.asarray(period)
                assert period.size == input_dim, "bad number of periods"
            else:
                period = np.ones(input_dim)

        if self.ARD2 == False:
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == 1, "Only one lengthscale needed for non-ARD kernel"
            else:
                lengthscale = np.ones(1)
        else:
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == input_dim, "bad number of lengthscales"
            else:
                lengthscale = np.ones(input_dim)

        self.variance = Param('variance', variance, Logexp())
        assert self.variance.size==1, "Variance size must be one"
        self.period =  Param('period', period, Logexp())
        self.lengthscale =  Param('lengthscale', lengthscale, Logexp())

        self.link_parameters(self.variance,  self.period, self.lengthscale)

    def parameters_changed(self):
        """
        This functions deals as a callback for each optimization iteration.
        If one optimization step was successfull and the parameters
        this callback function will be called to be able to update any
        precomputations for the kernel.
        """

        pass


    def K(self, X, X2=None):
        """Compute the covariance matrix between X and X2."""
        if X2 is None:
            X2 = X

        base = np.pi * (X[:, None, :] - X2[None, :, :]) / self.period
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale ), axis = -1 ) )

        return self.variance * exp_dist

    def Kdiag(self, X):
        """Compute the diagonal of the covariance matrix associated to X."""
        ret = np.empty(X.shape[0])
        ret[:] = self.variance
        return ret
      
    def dK_dX(self, X, X2=None):
      	lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
	periodinv = np.ones(X.shape[1])/(self.period)
	dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
	base = np.pi * dist * periodinv[:,None,None]
	exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
	return -self.variance*np.pi/2.*np.sin(2.*base)*exp_dist[None,:,:]*lengthscale2inv[:,None,None]*periodinv[:,None,None]
   
    def dK_dX2(self, X, X2=None):
	return -self.dK_dX(X, X2)
    
    def dK2_dXdX2(self, X, X2=None):
	lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
	periodinv = lengthscale2inv = np.ones(X.shape[1])/(self.period)
	dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
	base = np.pi * dist * periodinv[:,None,None]
	exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
	I = (np.ones((X.shape[0], X2.shape[0], X.shape[1], X2.shape[1]))*np.eye((X.shape[1]))).swapaxes(0,2).swapaxes(1,3)
	return - self.variance*((np.pi**2)/4*lengthscale2inv[:,None,None,None]*lengthscale2inv[None,:,None,None]*periodinv[:,None,None,None]*periodinv[None,:,None,None]*np.sin(2*base[:,None,:,:])*np.sin(2.*base[None,:,:,:])*exp_dist[None,None,:,:]+I*np.cos(2.*base[:,None,:,:])*exp_dist)
      
    def dK_dvariance(self, X, X2=None):
        if X2 is None:
            X2 = X
        base = np.pi * (X[:, None, :] - X2[None, :, :]) / self.period
        return np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale ), axis = -1 ) )
    
    def dK_dlengthscale(self, X, X2=None):
	if X2 is None:
	    X2=X
      	lengthscale3inv = np.ones(X.shape[1])/(self.lengthscale**3)
	periodinv = np.ones(X.shape[1])/(self.period)
	dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
	base = np.pi * dist *periodinv[:,None,None]
	exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
	return self.variance*np.sum((np.sin(base))**2, axis=0)*exp_dist*lengthscale3inv[:,None] if not self.ARD2 else self.variance*(np.sin(base))**2*exp_dist[None,:,:]*lengthscale3inv[:,None,None]
      
    def dK_dperiod(self, X, X2=None):
	if X2 is None:
	    X2=X
      	lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
	periodinv = np.ones(X.shape[1])/(self.period)
	dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
	base = np.pi * dist *periodinv[:,None,None]
	exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
	return self.variance*exp_dist*np.sum(np.pi/2.*dist*np.sin(2.*base), axis=0) if not self.ARD1 else self.variance*exp_dist[None,:,:]*np.pi/2.*dist*np.sin(2.*base)
      
    def dK2_dvariancedX(self, X, X2=None):
      	if X2 is None:
	    X2=X
	base = np.rollaxis(np.pi * (X[:, None, :] - X2[None, :, :]) / self.period ,2,0)
	exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
	lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
	periodinv = lengthscale2inv = np.ones(X.shape[1])/(self.period)
	return np.pi/2*np.sin(2.*base)*exp_dist[None,:,:]*lengthscale2inv[:,None,None]*periodinv[:,None,None]
    
    def dK2_dlengthscaledX(self, X, X2=None):
	if X2 is None:
	    X2=X
      	lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
      	lengthscale3inv = np.ones(X.shape[1])/(self.lengthscale**3)
	periodinv = np.ones(X.shape[1])/(self.period)
	dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
	base = np.pi * dist *periodinv[:,None,None]
	exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
	I = (np.ones((X.shape[0], X2.shape[0], X.shape[1], X2.shape[1]))*np.eye((X.shape[1]))).swapaxes(0,2).swapaxes(1,3)
	return np.pi*self.variance*lengthscale3inv[:,None,None]*periodinv[:,None,None]*np.sin(2.*base)*exp_dist[None,:,:]*(1-0.5*self.lengthscale*np.sum(lengthscale3inv[:,None,None]*np.sin(base)**2,  axis=0)) if not self.ARD2 else np.pi*self.variance*lengthscale3inv[:,None,None,None]*periodinv[None,:,None,None]*np.sin(2.*base[None,:,:,:])*exp_dist[None,None,:,:](I-0.5*(np.sin(base[:,None,:,:])**2)*lengthscale2inv[None,:,None,None])
      
    def dK2_dperioddX(self, X, X2=None):
	raise NotImplementedError
      
    def dK2_dvariancedX2(self, X, X2=None):
	raise -dK2_dvariancedX(self, X, X2)
    
    def dK2_dlengthscaledX2(self, X, X2=None):
	raise -dK2_dlengthscaledX(self, X, X2)
      
    def dK2_dperioddX2(self, X, X2=None):
	raise -dK2_dperioddX(self, X, X2)

    def dK3_dvariancedXdX2(self, X, X2=None):
	lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
	periodinv = lengthscale2inv = np.ones(X.shape[1])/(self.period)
	dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
	base = np.pi * dist * periodinv[:,None,None]
	exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
	I = (np.ones((X.shape[0], X2.shape[0], X.shape[1], X2.shape[1]))*np.eye((X.shape[1]))).swapaxes(0,2).swapaxes(1,3)
	return - (np.pi**2)/4*lengthscale2inv[:,None,None,None]*lengthscale2inv[None,:,None,None]*periodinv[:,None,None,None]*periodinv[None,:,None,None]*np.sin(2*base[:,None,:,:])*np.sin(2.*base[None,:,:,:])*exp_dist[None,None,:,:]+I*np.cos(2.*base[:,None,:,:])*exp_dist
    
    def dK3_dlengthscaledXdX2(self, X, X2=None):
	lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
	lengthscale3inv = np.ones(X.shape[1])/(self.lengthscale**3)
	periodinv = lengthscale2inv = np.ones(X.shape[1])/(self.period)
	dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
	base = np.pi * dist * periodinv[:,None,None]
	exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
	I = (np.ones((X.shape[0], X2.shape[0], X.shape[1], X2.shape[1]))*np.eye((X.shape[1]))).swapaxes(0,2).swapaxes(1,3)
	tmp1 = -0.25*self.variance*np.pi**2*lengthscale3inv[:,None,None,None,None]*lengthscale2inv[None,:,None,None,None]*lengthscale2inv[None,None,:,None,None]*(np.sin(base[:,None,None,:,:])**2)*np.sin(2.*base[None,:,None,:,:])*np.sin(2.*base[None,None,:,:,:])*exp_dist[None,None,None,:,:]
	# i = j
	tmp2 = 0.5*self.variance*lengthscale3inv[:,None,None,None,None]*lengthscale2inv[None,None,:,None,None]*periodinv[:,None,None,None,None]*periodinv[None,None,:,None,None]*I[:,:,None,:,:]*np.pi**2*np.sin(2.*base[:,None,None,:,:])*np.sin(2.*base[None,None,:,:,:])*exp_dist[None,None,None,:,:]
	# i = k
	tmp3 = 0.5*self.variance*lengthscale3inv[:,None,None,None,None]*lengthscale2inv[None,:,None,None,None]*periodinv[:,None,None,None,None]*periodinv[None,:,None,None,None]*I[:,None,:,:,:]*np.pi**2*np.sin(2.*base[:,None,None,:,:])*np.sin(2.*base[None,:,None,:,:])*exp_dist[None,None,None,:,:]
	# j = k
	tmp4 = I[None,:,:,:,:]
	# i=j=k
	tmp5 = I[None,:,:,:,:]*I[:,None,:,:,:]
	return tmp1+tmp2+tmp3+tmp4+tmp5
      
    def dK3_dperioddXdX2(self, X, X2=None):
	raise NotImplementedError

    def update_gradients_full(self, dL_dK, X, X2=None):
        """derivative of the covariance matrix with respect to the parameters."""
        if X2 is None:
            X2 = X

        base = np.pi * (X[:, None, :] - X2[None, :, :]) / self.period

        sin_base = np.sin( base )
        exp_dist = np.exp( -0.5* np.sum( np.square(  sin_base / self.lengthscale ), axis = -1 ) )

        dwl = self.variance * (1.0/np.square(self.lengthscale)) * sin_base*np.cos(base) * (base / self.period)

        dl = self.variance * np.square( sin_base) / np.power( self.lengthscale, 3)

        self.variance.gradient = np.sum(exp_dist * dL_dK)
        #target[0] += np.sum( exp_dist * dL_dK)

        if self.ARD1: # different periods
            self.period.gradient = (dwl * exp_dist[:,:,None] * dL_dK[:, :, None]).sum(0).sum(0)
        else:  # same period
            self.period.gradient = np.sum(dwl.sum(-1) * exp_dist * dL_dK)

        if self.ARD2: # different lengthscales
            self.lengthscale.gradient = (dl * exp_dist[:,:,None] * dL_dK[:, :, None]).sum(0).sum(0)
        else: # same lengthscales
            self.lengthscale.gradient = np.sum(dl.sum(-1) * exp_dist * dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        """derivative of the diagonal of the covariance matrix with respect to the parameters."""
        self.variance.gradient = np.sum(dL_dKdiag)
        self.period.gradient = 0
        self.lengthscale.gradient = 0

    def gradients_X(self, dL_dK, X, X2=None):
        K = self.K(X, X2)
        if X2 is None:
            dL_dK = dL_dK+dL_dK.T
            X2 = X
        dX = -np.pi*((dL_dK*K)[:,:,None]*np.sin(2*np.pi/self.period*(X[:,None,:] - X2[None,:,:]))/(2.*np.square(self.lengthscale)*self.period)).sum(1)
        return dX
    
    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)
    
    def input_sensitivity(self, summarize=True):
        return self.variance*np.ones(self.input_dim)/self.lengthscale**2