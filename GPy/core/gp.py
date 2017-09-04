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

import logging
import warnings
logger = logging.getLogger("GP")

class GP(Model):
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
    def __init__(self, X, Y, kernel, likelihood, mean_function=None, inference_method=None, name='gp', Y_metadata=None, normalizer=False):
        super(GP, self).__init__(name)

        #assert X.ndim == 2
        if isinstance(X, (ObsAr, VariationalPosterior)):
            self.X = X.copy()
        else: self.X = ObsAr(X)

        self.num_data, self.input_dim = self.X.shape

        assert Y.ndim == 2
        logger.info("initializing Y")

        if normalizer is True:
            self.normalizer = Standardize()
        elif normalizer is False:
            self.normalizer = None
        else:
            self.normalizer = normalizer

        if self.normalizer is not None:
            self.normalizer.scale_by(Y)
            self.Y_normalized = ObsAr(self.normalizer.normalize(Y))
            self.Y = Y
        elif isinstance(Y, np.ndarray):
            self.Y = ObsAr(Y)
            self.Y_normalized = self.Y
        else:
            self.Y = Y
            self.Y_normalized = self.Y

        if Y.shape[0] != self.num_data:
            #There can be cases where we want inputs than outputs, for example if we have multiple latent
            #function values
            warnings.warn("There are more rows in your input data X, \
                         than in your output data Y, be VERY sure this is what you want")
        _, self.output_dim = self.Y.shape

        assert ((Y_metadata is None) or isinstance(Y_metadata, dict))
        self.Y_metadata = Y_metadata

        assert isinstance(kernel, kern.Kern)
        #assert self.input_dim == kernel.input_dim
        self.kern = kernel

        assert isinstance(likelihood, likelihoods.Likelihood)
        self.likelihood = likelihood

        if self.kern._effective_input_dim != self.X.shape[1]:
            warnings.warn("Your kernel has a different input dimension {} then the given X dimension {}. Be very sure this is what you want and you have not forgotten to set the right input dimenion in your kernel".format(self.kern._effective_input_dim, self.X.shape[1]))

        #handle the mean function
        self.mean_function = mean_function
        if mean_function is not None:
            assert isinstance(self.mean_function, Mapping)
            assert mean_function.input_dim == self.input_dim
            assert mean_function.output_dim == self.output_dim
            self.link_parameter(mean_function)

        #find a sensible inference method
        logger.info("initializing inference method")
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian) or isinstance(likelihood, likelihoods.MixedNoise):
                inference_method = exact_gaussian_inference.ExactGaussianInference()
            else:
                inference_method = expectation_propagation.EP()
                print("defaulting to " + str(inference_method) + " for latent function inference")
        self.inference_method = inference_method

        logger.info("adding kernel and likelihood as parameters")
        self.link_parameter(self.kern)
        self.link_parameter(self.likelihood)
        self.posterior = None


        # The predictive variable to be used to predict using the posterior object's
        # woodbury_vector and woodbury_inv is defined as predictive_variable
        # as long as the posterior has the right woodbury entries.
        # It is the input variable used for the covariance between
        # X_star and the posterior of the GP.
        # This is usually just a link to self.X (full GP) or self.Z (sparse GP).
        # Make sure to name this variable and the predict functions will "just work"
        # In maths the predictive variable is:
        #         K_{xx} - K_{xp}W_{pp}^{-1}K_{px}
        #         W_{pp} := \texttt{Woodbury inv}
        #         p := _predictive_variable

    @property
    def _predictive_variable(self):
        return self.X

    def set_XY(self, X=None, Y=None):
        """
        Set the input / output data of the model
        This is useful if we wish to change our existing data but maintain the same model

        :param X: input observations
        :type X: np.ndarray
        :param Y: output observations
        :type Y: np.ndarray
        """
        self.update_model(False)
        if Y is not None:
            if self.normalizer is not None:
                self.normalizer.scale_by(Y)
                self.Y_normalized = ObsAr(self.normalizer.normalize(Y))
                self.Y = Y
            else:
                self.Y = ObsAr(Y)
                self.Y_normalized = self.Y
        if X is not None:
            if self.X in self.parameters:
                # LVM models
                if isinstance(self.X, VariationalPosterior):
                    assert isinstance(X, type(self.X)), "The given X must have the same type as the X in the model!"
                    index = self.X._parent_index_
                    self.unlink_parameter(self.X)
                    self.X = X
                    self.link_parameter(self.X, index=index)
                else:
                    index = self.X._parent_index_
                    self.unlink_parameter(self.X)
                    from ..core import Param
                    self.X = Param('latent mean', X)
                    self.link_parameter(self.X, index=index)
            else:
                self.X = ObsAr(X)
        self.update_model(True)

    def set_X(self,X):
        """
        Set the input data of the model

        :param X: input observations
        :type X: np.ndarray
        """
        self.set_XY(X=X)

    def set_Y(self,Y):
        """
        Set the output data of the model

        :param X: output observations
        :type X: np.ndarray
        """
        self.set_XY(Y=Y)

    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y_normalized, self.mean_function, self.Y_metadata)
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)
        if self.mean_function is not None:
            self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)

    def log_likelihood(self):
        """
        The log marginal likelihood of the model, :math:`p(\mathbf{y})`, this is the objective function of the model being optimised
        """
        return self._log_marginal_likelihood

    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        """
        For making predictions, does not account for normalization or likelihood

        full_cov is a boolean which defines whether the full covariance matrix
        of the prediction is computed. If full_cov is False (default), only the
        diagonal of the covariance is returned.

        .. math::
            p(f*|X*, X, Y) = \int^{\inf}_{\inf} p(f*|f,X*)p(f|X,Y) df
                        = N(f*| K_{x*x}(K_{xx} + \Sigma)^{-1}Y, K_{x*x*} - K_{xx*}(K_{xx} + \Sigma)^{-1}K_{xx*}
            \Sigma := \texttt{Likelihood.variance / Approximate likelihood covariance}
        """
        mu, var = self.posterior._raw_predict(kern=self.kern if kern is None else kern, Xnew=Xnew, pred_var=self._predictive_variable, full_cov=full_cov)
        if self.mean_function is not None:
            mu += self.mean_function.f(Xnew)
        return mu, var

    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True):
        """
        Predict the function(s) at the new point(s) Xnew. This includes the likelihood
        variance added to the predicted underlying function (usually referred to as f).

        In order to predict without adding in the likelihood give
        `include_likelihood=False`, or refer to self.predict_noiseless().

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray (Nnew x self.input_dim)
        :param full_cov: whether to return the full covariance matrix, or just
                         the diagonal
        :type full_cov: bool
        :param Y_metadata: metadata about the predicting point to pass to the likelihood
        :param kern: The kernel to use for prediction (defaults to the model
                     kern). this is useful for examining e.g. subprocesses.
        :param bool include_likelihood: Whether or not to add likelihood noise to the predicted underlying latent function f.

        :returns: (mean, var):
            mean: posterior mean, a Numpy array, Nnew x self.input_dim
            var: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise

           If full_cov and self.input_dim > 1, the return shape of var is Nnew x Nnew x self.input_dim. If self.input_dim == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.

        Note: If you want the predictive quantiles (e.g. 95% confidence interval) use :py:func:"~GPy.core.gp.GP.predict_quantiles".
        """
        #predict the latent function values
        mu, var = self._raw_predict(Xnew, full_cov=full_cov, kern=kern)

        if include_likelihood:
            # now push through likelihood
            if likelihood is None:
                likelihood = self.likelihood
            mu, var = likelihood.predictive_values(mu, var, full_cov, Y_metadata=Y_metadata)

        if self.normalizer is not None:
            mu, var = self.normalizer.inverse_mean(mu), self.normalizer.inverse_variance(var)

        return mu, var

    def predict_noiseless(self,  Xnew, full_cov=False, Y_metadata=None, kern=None):
        """
        Convenience function to predict the underlying function of the GP (often
        referred to as f) without adding the likelihood variance on the
        prediction function.

        This is most likely what you want to use for your predictions.

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray (Nnew x self.input_dim)
        :param full_cov: whether to return the full covariance matrix, or just
                         the diagonal
        :type full_cov: bool
        :param Y_metadata: metadata about the predicting point to pass to the likelihood
        :param kern: The kernel to use for prediction (defaults to the model
                     kern). this is useful for examining e.g. subprocesses.

        :returns: (mean, var):
            mean: posterior mean, a Numpy array, Nnew x self.input_dim
            var: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise

           If full_cov and self.input_dim > 1, the return shape of var is Nnew x Nnew x self.input_dim. If self.input_dim == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.

        Note: If you want the predictive quantiles (e.g. 95% confidence interval) use :py:func:"~GPy.core.gp.GP.predict_quantiles".
        """
        return self.predict(Xnew, full_cov, Y_metadata, kern, None, False)

    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None, kern=None, likelihood=None):
        """
        Get the predictive quantiles around the prediction at X

        :param X: The points at which to make a prediction
        :type X: np.ndarray (Xnew x self.input_dim)
        :param quantiles: tuple of quantiles, default is (2.5, 97.5) which is the 95% interval
        :type quantiles: tuple
        :param kern: optional kernel to use for prediction
        :type predict_kw: dict
        :returns: list of quantiles for each X and predictive quantiles for interval combination
        :rtype: [np.ndarray (Xnew x self.output_dim), np.ndarray (Xnew x self.output_dim)]
        """
        m, v = self._raw_predict(X,  full_cov=False, kern=kern)
        if likelihood is None:
            likelihood = self.likelihood
            
        quantiles = likelihood.predictive_quantiles(m, v, quantiles, Y_metadata=Y_metadata)
        
        if self.normalizer is not None:
            quantiles = [self.normalizer.inverse_mean(q) for q in quantiles]
        return quantiles

    def predictive_gradients(self, Xnew, kern=None):
        """
        Compute the derivatives of the predicted latent function with respect to X*

        Given a set of points at which to predict X* (size [N*,Q]), compute the
        derivatives of the mean and variance. Resulting arrays are sized:
         dmu_dX* -- [N*, Q ,D], where D is the number of output in this GP (usually one).

        Note that this is not the same as computing the mean and variance of the derivative of the function!

         dv_dX*  -- [N*, Q],    (since all outputs have the same variance)
        :param X: The points at which to get the predictive gradients
        :type X: np.ndarray (Xnew x self.input_dim)
        :returns: dmu_dX, dv_dX
        :rtype: [np.ndarray (N*, Q ,D), np.ndarray (N*,Q) ]

        """
        if kern is None:
            kern = self.kern
        mean_jac = np.empty((Xnew.shape[0],Xnew.shape[1],self.output_dim))

        for i in range(self.output_dim):
            mean_jac[:,:,i] = kern.gradients_X(self.posterior.woodbury_vector[:,i:i+1].T, Xnew, self._predictive_variable)

        # gradients wrt the diagonal part k_{xx}
        dv_dX = kern.gradients_X(np.eye(Xnew.shape[0]), Xnew)
        #grads wrt 'Schur' part K_{xf}K_{ff}^{-1}K_{fx}
        if self.posterior.woodbury_inv.ndim == 3:
            tmp = np.empty(dv_dX.shape + (self.posterior.woodbury_inv.shape[2],))
            tmp[:] = dv_dX[:,:,None]
            for i in range(self.posterior.woodbury_inv.shape[2]):
                alpha = -2.*np.dot(kern.K(Xnew, self._predictive_variable), self.posterior.woodbury_inv[:, :, i])
                tmp[:, :, i] += kern.gradients_X(alpha, Xnew, self._predictive_variable)
        else:
            tmp = dv_dX
            alpha = -2.*np.dot(kern.K(Xnew, self._predictive_variable), self.posterior.woodbury_inv)
            tmp += kern.gradients_X(alpha, Xnew, self._predictive_variable)
        return mean_jac, tmp

    def predict_jacobian(self, Xnew, kern=None, full_cov=False):
        """
        Compute the derivatives of the posterior of the GP.

        Given a set of points at which to predict X* (size [N*,Q]), compute the
        mean and variance of the derivative. Resulting arrays are sized:

         dL_dX* -- [N*, Q ,D], where D is the number of output in this GP (usually one).
          Note that this is the mean and variance of the derivative,
          not the derivative of the mean and variance! (See predictive_gradients for that)

         dv_dX*  -- [N*, Q],    (since all outputs have the same variance)
          If there is missing data, it is not implemented for now, but
          there will be one output variance per output dimension.

        :param X: The points at which to get the predictive gradients.
        :type X: np.ndarray (Xnew x self.input_dim)
        :param kern: The kernel to compute the jacobian for.
        :param boolean full_cov: whether to return the cross-covariance terms between
        the N* Jacobian vectors

        :returns: dmu_dX, dv_dX
        :rtype: [np.ndarray (N*, Q ,D), np.ndarray (N*,Q,(D)) ]
        """
        if kern is None:
            kern = self.kern

        mean_jac = np.empty((Xnew.shape[0],Xnew.shape[1],self.output_dim))

        for i in range(self.output_dim):
            mean_jac[:,:,i] = kern.gradients_X(self.posterior.woodbury_vector[:,i:i+1].T, Xnew, self._predictive_variable)

        dK_dXnew_full = np.empty((self._predictive_variable.shape[0], Xnew.shape[0], Xnew.shape[1]))
        one = np.ones((1,1))
        for i in range(self._predictive_variable.shape[0]):
            dK_dXnew_full[i] = kern.gradients_X(one, Xnew, self._predictive_variable[[i]])

        if full_cov:
            dK2_dXdX = kern.gradients_XX(one, Xnew)
        else:
            dK2_dXdX = kern.gradients_XX_diag(one, Xnew)
            #dK2_dXdX = np.zeros((Xnew.shape[0], Xnew.shape[1], Xnew.shape[1]))
            #for i in range(Xnew.shape[0]):
            #    dK2_dXdX[i:i+1,:,:] = kern.gradients_XX(one, Xnew[i:i+1,:])

        def compute_cov_inner(wi):
            if full_cov:
                var_jac = dK2_dXdX - np.einsum('qnm,msr->nsqr', dK_dXnew_full.T.dot(wi), dK_dXnew_full) # n,s = Xnew.shape[0], m = pred_var.shape[0]
            else:
                var_jac = dK2_dXdX - np.einsum('qnm,mnr->nqr', dK_dXnew_full.T.dot(wi), dK_dXnew_full)
            return var_jac

        if self.posterior.woodbury_inv.ndim == 3: # Missing data:
            if full_cov:
                var_jac = np.empty((Xnew.shape[0],Xnew.shape[0],Xnew.shape[1],Xnew.shape[1],self.output_dim))
                for d in range(self.posterior.woodbury_inv.shape[2]):
                    var_jac[:, :, :, :, d] = compute_cov_inner(self.posterior.woodbury_inv[:, :, d])
            else:
                var_jac = np.empty((Xnew.shape[0],Xnew.shape[1],Xnew.shape[1],self.output_dim))
                for d in range(self.posterior.woodbury_inv.shape[2]):
                    var_jac[:, :, :, d] = compute_cov_inner(self.posterior.woodbury_inv[:, :, d])
        else:
            var_jac = compute_cov_inner(self.posterior.woodbury_inv)
        return mean_jac, var_jac

    def predict_wishart_embedding(self, Xnew, kern=None, mean=True, covariance=True):
        """
        Predict the wishart embedding G of the GP. This is the density of the
        input of the GP defined by the probabilistic function mapping f.
        G = J_mean.T*J_mean + output_dim*J_cov.

        :param array-like Xnew: The points at which to evaluate the magnification.
        :param :py:class:`~GPy.kern.Kern` kern: The kernel to use for the magnification.

        Supplying only a part of the learning kernel gives insights into the density
        of the specific kernel part of the input function. E.g. one can see how dense the
        linear part of a kernel is compared to the non-linear part etc.
        """
        if kern is None:
            kern = self.kern

        mu_jac, var_jac = self.predict_jacobian(Xnew, kern, full_cov=False)
        mumuT = np.einsum('iqd,ipd->iqp', mu_jac, mu_jac)
        Sigma = np.zeros(mumuT.shape)
        if var_jac.ndim == 4: # Missing data
            Sigma = var_jac.sum(-1)
        else:
            Sigma = self.output_dim*var_jac

        G = 0.
        if mean:
            G += mumuT
        if covariance:
            G += Sigma
        return G

    def predict_wishard_embedding(self, Xnew, kern=None, mean=True, covariance=True):
        warnings.warn("Wrong naming, use predict_wishart_embedding instead. Will be removed in future versions!", DeprecationWarning)
        return self.predict_wishart_embedding(Xnew, kern, mean, covariance)

    def predict_magnification(self, Xnew, kern=None, mean=True, covariance=True, dimensions=None):
        """
        Predict the magnification factor as

        sqrt(det(G))

        for each point N in Xnew.

        :param bool mean: whether to include the mean of the wishart embedding.
        :param bool covariance: whether to include the covariance of the wishart embedding.
        :param array-like dimensions: which dimensions of the input space to use [defaults to self.get_most_significant_input_dimensions()[:2]]
        """
        G = self.predict_wishart_embedding(Xnew, kern, mean, covariance)
        if dimensions is None:
            dimensions = self.get_most_significant_input_dimensions()[:2]
        G = G[:, dimensions][:,:,dimensions]
        from ..util.linalg import jitchol
        mag = np.empty(Xnew.shape[0])
        for n in range(Xnew.shape[0]):
            try:
                mag[n] = np.sqrt(np.exp(2*np.sum(np.log(np.diag(jitchol(G[n, :, :]))))))
            except:
                mag[n] = np.sqrt(np.linalg.det(G[n, :, :]))
        return mag

    def posterior_samples_f(self,X, size=10, full_cov=True, **predict_kwargs):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param full_cov: whether to return the full covariance matrix, or just the diagonal.
        :type full_cov: bool.
        :returns: fsim: set of simulations
        :rtype: np.ndarray (D x N x samples) (if D==1 we flatten out the first dimension)
        """
        m, v = self._raw_predict(X,  full_cov=full_cov, **predict_kwargs)
        if self.normalizer is not None:
            m, v = self.normalizer.inverse_mean(m), self.normalizer.inverse_variance(v)

        def sim_one_dim(m, v):
            if not full_cov:
                return np.random.multivariate_normal(m.flatten(), np.diag(v.flatten()), size).T
            else:
                return np.random.multivariate_normal(m.flatten(), v, size).T

        if self.output_dim == 1:
            return sim_one_dim(m, v)
        else:
            fsim = np.empty((self.output_dim, self.num_data, size))
            for d in range(self.output_dim):
                if full_cov and v.ndim == 3:
                    fsim[d] = sim_one_dim(m[:, d], v[:, :, d])
                elif (not full_cov) and v.ndim == 2:
                    fsim[d] = sim_one_dim(m[:, d], v[:, d])
                else:
                    fsim[d] = sim_one_dim(m[:, d], v)
        return fsim

    def posterior_samples(self, X, size=10, full_cov=False, Y_metadata=None, likelihood=None, **predict_kwargs):
        """
        Samples the posterior GP at the points X.

        :param X: the points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim.)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param full_cov: whether to return the full covariance matrix, or just the diagonal.
        :type full_cov: bool.
        :param noise_model: for mixed noise likelihood, the noise model to use in the samples.
        :type noise_model: integer.
        :returns: Ysim: set of simulations,
        :rtype: np.ndarray (D x N x samples) (if D==1 we flatten out the first dimension)
        """
        fsim = self.posterior_samples_f(X, size, full_cov=full_cov, **predict_kwargs)
        if likelihood is None:
            likelihood = self.likelihood
        if fsim.ndim == 3:
            for d in range(fsim.shape[0]):
                fsim[d] = likelihood.samples(fsim[d], Y_metadata=Y_metadata)
        else:
            fsim = likelihood.samples(fsim, Y_metadata=Y_metadata)
        return fsim

    def input_sensitivity(self, summarize=True):
        """
        Returns the sensitivity for each dimension of this model
        """
        return self.kern.input_sensitivity(summarize=summarize)

    def get_most_significant_input_dimensions(self, which_indices=None):
        return self.kern.get_most_significant_input_dimensions(which_indices)

    def optimize(self, optimizer=None, start=None, messages=False, max_iters=1000, ipython_notebook=True, clear_after_finish=False, **kwargs):
        """
        Optimize the model using self.log_likelihood and self.log_likelihood_gradient, as well as self.priors.
        kwargs are passed to the optimizer. They can be:

        :param max_iters: maximum number of function evaluations
        :type max_iters: int
        :messages: whether to display during optimisation
        :type messages: bool
        :param optimizer: which optimizer to use (defaults to self.preferred optimizer), a range of optimisers can be found in :module:`~GPy.inference.optimization`, they include 'scg', 'lbfgs', 'tnc'.
        :type optimizer: string
        :param bool ipython_notebook: whether to use ipython notebook widgets or not.
        :param bool clear_after_finish: if in ipython notebook, we can clear the widgets after optimization.
        """
        self.inference_method.on_optimization_start()
        try:
            ret = super(GP, self).optimize(optimizer, start, messages, max_iters, ipython_notebook, clear_after_finish, **kwargs)
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught, calling on_optimization_end() to round things up")
            self.inference_method.on_optimization_end()
            raise
        return ret

    def infer_newX(self, Y_new, optimize=True):
        """
        Infer X for the new observed data *Y_new*.

        :param Y_new: the new observed data for inference
        :type Y_new: numpy.ndarray
        :param optimize: whether to optimize the location of new X (True by default)
        :type optimize: boolean
        :return: a tuple containing the posterior estimation of X and the model that optimize X
        :rtype: (:class:`~GPy.core.parameterization.variational.VariationalPosterior` and numpy.ndarray, :class:`~GPy.core.model.Model`)
        """
        from ..inference.latent_function_inference.inferenceX import infer_newX
        return infer_newX(self, Y_new, optimize=optimize)

    def log_predictive_density(self, x_test, y_test, Y_metadata=None):
        """
        Calculation of the log predictive density

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param x_test: test locations (x_{*})
        :type x_test: (Nx1) array
        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param Y_metadata: metadata associated with the test points
        """
        mu_star, var_star = self._raw_predict(x_test)
        return self.likelihood.log_predictive_density(y_test, mu_star, var_star, Y_metadata=Y_metadata)

    def log_predictive_density_sampling(self, x_test, y_test, Y_metadata=None, num_samples=1000):
        """
        Calculation of the log predictive density by sampling

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param x_test: test locations (x_{*})
        :type x_test: (Nx1) array
        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param Y_metadata: metadata associated with the test points
        :param num_samples: number of samples to use in monte carlo integration
        :type num_samples: int
        """
        mu_star, var_star = self._raw_predict(x_test)
        return self.likelihood.log_predictive_density_sampling(y_test, mu_star, var_star, Y_metadata=Y_metadata, num_samples=num_samples)
    
    def check_gradients(self, df=1e-7):
        x = self.optimizer_array.copy()
        analytic = self._log_likelihood_gradients()
        num = [0 in range(len(analytic))]
        log_orig = self.log_likelihood()
        x = self.optimizer_array.copy()
        for i in range(len(x)):
            x[i]._update_on = False
            t = x.values
            x[i] += df
            log_new = self.log_likelihood()
            num[i] = (log_new-log_orig)/d 
            x[i].values = t
            x[i]._update_on = True
        print(num-analytic)
        
    def checkgrad(self, target_param=None, verbose=False, step=1e-6, tolerance=1e-3, df_tolerance=1e-12):
        """
        Check the gradient of the ,odel by comparing to a numerical
        estimate.  If the verbose flag is passed, individual
        components are tested (and printed)
        :param verbose: If True, print a "full" checking of each parameter
        :type verbose: bool
        :param step: The size of the step around which to linearise the objective
        :type step: float (default 1e-6)
        :param tolerance: the tolerance allowed (see note)
        :type tolerance: float (default 1e-3)
        Note:-
            The gradient is considered correct if the ratio of the analytical
            and numerical gradients is within <tolerance> of unity.
            The *dF_ratio* indicates the limit of numerical accuracy of numerical gradients.
            If it is too small, e.g., smaller than 1e-12, the numerical gradients are usually
            not accurate enough for the tests (shown with blue).
        """
        if not self._model_initialized_:
            import warnings
            warnings.warn("This model has not been initialized, try model.inititialize_model()", RuntimeWarning)
            return False

        x = self.optimizer_array.copy()

        if not verbose:
            # make sure only to test the selected parameters
            if target_param is None:
                transformed_index = np.arange(len(x))
            else:
                transformed_index = self._raveled_index_for_transformed(target_param)

                if transformed_index.size == 0:
                    print("No free parameters to check")
                    return True

            # just check the global ratio
            dx = np.zeros(x.shape)
            dx[transformed_index] = step * (np.sign(np.random.uniform(-1, 1, transformed_index.size)) if transformed_index.size != 2 else 1.)

            # evaulate around the point x
            f1 = self._objective(x + dx)
            f2 = self._objective(x - dx)
            gradient = self._grads(x)

            dx = dx[transformed_index]
            gradient = gradient[transformed_index]

            denominator = (2 * np.dot(dx, gradient))
            global_ratio = (f1 - f2) / np.where(denominator == 0., 1e-32, denominator)
            global_diff = np.abs(f1 - f2) < tolerance and np.allclose(gradient, 0, atol=tolerance)
            if global_ratio is np.nan: # pragma: no cover
                global_ratio = 0
            return np.abs(1. - global_ratio) < tolerance or global_diff
        else:
            # check the gradient of each parameter individually, and do some pretty printing
            try:
                names = self.parameter_names_flat()
            except NotImplementedError:
                names = ['Variable %i' % i for i in range(len(x))]
            # Prepare for pretty-printing
            header = ['Name', 'Ratio', 'Difference', 'Analytical', 'Numerical', 'dF_ratio']
            max_names = max([len(names[i]) for i in range(len(names))] + [len(header[0])])
            float_len = 10
            cols = [max_names]
            cols.extend([max(float_len, len(header[i])) for i in range(1, len(header))])
            cols = np.array(cols) + 5
            header_string = ["{h:^{col}}".format(h=header[i], col=cols[i]) for i in range(len(cols))]
            header_string = list(map(lambda x: '|'.join(x), [header_string]))
            separator = '-' * len(header_string[0])
            print('\n'.join([header_string[0], separator]))

            if target_param is None:
                target_param = self
            transformed_index = self._raveled_index_for_transformed(target_param)

            if transformed_index.size == 0:
                print("No free parameters to check")
                return True

            gradient = self._grads(x).copy()
            np.where(gradient == 0, 1e-312, gradient)
            ret = True
            for xind in zip(transformed_index):
                xx = x.copy()
                xx[xind] += step
                f1 = float(self._objective(xx))
                xx[xind] -= 2.*step
                f2 = float(self._objective(xx))
                #Avoid divide by zero, if any of the values are above 1e-15, otherwise both values are essentiall
                #the same
                if f1 > 1e-15 or f1 < -1e-15 or f2 > 1e-15 or f2 < -1e-15:
                    df_ratio = np.abs((f1 - f2) / min(f1, f2))
                else: # pragma: no cover
                    df_ratio = 1.0
                df_unstable = df_ratio < df_tolerance
                numerical_gradient = (f1 - f2) / (2. * step)
                if np.all(gradient[xind] == 0): # pragma: no cover
                    ratio = (f1 - f2) == gradient[xind]
                else:
                    ratio = (f1 - f2) / (2. * step * gradient[xind])
                difference = np.abs(numerical_gradient - gradient[xind])

                if (np.abs(1. - ratio) < tolerance) or np.abs(difference) < tolerance:
                    formatted_name = "\033[92m {0} \033[0m".format(names[xind])
                    ret &= True
                else:  # pragma: no cover
                    formatted_name = "\033[91m {0} \033[0m".format(names[xind])
                    ret &= False
                if df_unstable:  # pragma: no cover
                    formatted_name = "\033[94m {0} \033[0m".format(names[xind])

                r = '%.6f' % float(ratio)
                d = '%.6f' % float(difference)
                g = '%.6f' % gradient[xind]
                ng = '%.6f' % float(numerical_gradient)
                df = '%1.e' % float(df_ratio)
                grad_string = "{0:<{c0}}|{1:^{c1}}|{2:^{c2}}|{3:^{c3}}|{4:^{c4}}|{5:^{c5}}".format(formatted_name, r, d, g, ng, df, c0=cols[0] + 9, c1=cols[1], c2=cols[2], c3=cols[3], c4=cols[4], c5=cols[5])
                print(grad_string)

            self.optimizer_array = x
            return ret