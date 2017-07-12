from .kern import CombinationKernel
import numpy as np
from paramz.caching import Cache_this

# A thin wrapper around the base kernel to tell that we are dealing with a partial derivative of a Kernel
class DiffKern(CombinationKernel):
    def __init__(self, base_kern, dimension):
        super(DiffKern, self).__init__([base_kern], 'diffKern')
        self.base_kern = base_kern
        self.dimension = dimension

    def parameters_changed(self):
        self.base_kern.parameters_changed()

    @Cache_this(limit=3, ignore_args=())
    def K(self, X, X2, dimX2 = None): #X in dimension self.dimension
        if dimX2 is None:
            dimX2 = self.dimension
        return np.reshape(self.base_kern.dK2_dXdX2(X,X2)[self.dimension, dimX2,:,:], (X.shape[0], X2.shape[0]))
 
    @Cache_this(limit=3, ignore_args=())
    def Kdiag(self, X):
        return np.diag(self.base_kern.dK2_dXdX2(X,X)[self.dimension, self.dimension, :, :])
    
    @Cache_this(limit=3, ignore_args=())
    def dK_dX(self, X, X2): #X in dimension self.dimension
        return np.reshape(self.base_kern.dK_dX(X,X2)[self.dimension,:,:], (X.shape[0], X2.shape[0]))
    
    def reset_gradients(self):
        self.base_kern.reset_gradients()
    
    @Cache_this(limit=3, ignore_args=())
    def dK_dX2(self, X, X2): #X in dimension self.dimension
        return np.reshape(self.base_kern.dK_dX2(X,X2)[self.dimension,:,:], (X.shape[0], X2.shape[0]))
    
    def update_gradients_full(self, dL_dK, X, X2=None, reset=True, dimX2=None):
        if dimX2 is None:
            dimX2 = self.dimension
        gradients = self.base_kern.dgradients2_dXdX2(X,X2)
        #print("update_gradients_full, {} {}".format(self.dimension, dimX2))
        #print(gradients[3][self.dimension,dimX2,:,:])
        self.base_kern.update_gradients_direct([np.sum(dL_dK*gradient[self.dimension,dimX2,:,:]) for gradient in gradients], reset)

    def update_gradients_diag(self, dL_dK_diag, X, reset=True): #X in dimension self.dimension
        gradients = self.base_kern.dgradients2_dXdX2(X,X)
        self.base_kern.update_gradients_direct([np.sum(dL_dK_diag*np.diag(gradient[self.dimension, self.dimension,:,:])) for gradient in gradients], reset)
    
    def update_gradients_dK_dX(self, dL_dK, X, X2=None, reset=True): #X in dimension self.dimension
        gradients = self.base_kern.dgradients_dX(X,X2)
        #print("update_gradients_dK_dX, {}".format(self.dimension))
        #print(gradients[3][self.dimension,:,:])
        self.base_kern.update_gradients_direct([np.sum(dL_dK*gradient[self.dimension,:,:]) for gradient in gradients], reset)
        
    def update_gradients_dK_dX2(self, dL_dK, X, X2=None, reset=True): #X in dimension self.dimension
        gradients = self.base_kern.dgradients_dX2(X,X2)
        #print("update_gradients_dK_dX2, {}".format(self.dimension))
        #print(gradients[3][self.dimension,:,:])
        self.base_kern.update_gradients_direct([np.sum(dL_dK*gradient[self.dimension,:,:]) for gradient in gradients], reset)