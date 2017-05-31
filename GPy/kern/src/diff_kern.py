from .kern import CombinationKernel
import numpy as np

# A thin wrapper around the base kernel to tell that we are dealing with a partial derivative of a Kernel
class DiffKern(CombinationKernel):
    def __init__(self, base_kern, dimension):
        super(DiffKern, self).__init__([base_kern], 'diffKern')
        self.base_kern = base_kern
        self.dimension = dimension

    def parameters_changed(self):
        self.base_kern.parameters_changed()

    def K(self, X, X2, dimX2 = None): #X in dimension self.dimension
        if dimX2 is None:
            dimX2 = self.dimension
        return self.base_kern.dK2_dXdX2(X,X2)[self.dimension, dimX2,:,:]
 
    def Kdiag(self, X):
        return np.diag(self.base_kern.dK2_dXdX2(X,X2)[self.dimension, dimX2,:,:])
    
    def dK_dX(self, X, X2): #X in dimension self.dimension
        return self.base_kern.dK_dX(X,X2)[self.dimension,:,:]
    
    def dK_dX2(self, X, X2): #X in dimension self.dimension
        return self.base_kern.dK_dX2(X,X2)[self.dimension,:,:]
    
    def update_gradients_full(self, dL_dK, X, X2=None, dimX2=None, reset=True):
        if dimX2 is None:
            dimX2 = self.dimension
        gradients = self.base_kern.dgradients2_dXdX2(X,X2)
        self.base_kern.update_gradients_direct(self, [dL_dK*gradient[self.dimension,dimX2,:,:] for gradient in grandients], reset)

    def update_gradients_diag(self, dL_dK_diag, X, reset=True): #X in dimension self.dimension
        gradients = self.base_kern.dgradients2_dXdX2(X,X)
        self.base_kern.update_gradients_direct(self, [dL_dK_diag*np.diag(gradient[self.dimension, self.dimension,:,:]) for gradient in grandients], reset)
    
    def update_gradients_dK_dX(self, dL_dK, X, X2=None, reset=True): #X in dimension self.dimension
        gradients = self.base_kern.dgradients_dX(X,X2)
        self.base_kern.update_gradients_direct(self, [dL_dK*gradient[self.dimension,:,:] for gradient in grandients], reset)
        
    def update_gradients_dK_dX2(self, dL_dK, X, X2=None, reset=True): #X in dimension self.dimension
        gradients = self.base_kern.dgradients_dX2(X,X2)
        self.base_kern.update_gradients_direct(self, [dL_dK*gradient[self.dimension,:,:] for gradient in grandients], reset)