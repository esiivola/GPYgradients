from .kern import Kern, CombinationKernel
import numpy as np
from functools import reduce, partial
#from .independent_outputs import index_to_slices
from GPy.util.multioutput import index_to_slices
from paramz.caching import Cache_this
    
class MultioutputKern(CombinationKernel):
    def __init__(self, kernels, cross_covariances={}, name='MultioutputKern'):
        if not isinstance(kernels, list):
            self.single_kern = True
            self.kern = kernels
            kernels = [kernels]
        else:
            self.single_kern = False
            self.kern = kernels
        # The combination kernel ALLWAYS puts the extra dimension last.
        # Thus, the index dimension of this kernel is always the last dimension
        # after slicing. This is why the index_dim is just the last column:
        self.index_dim = -1
        super(MultioutputKern, self).__init__(kernels=kernels, extra_dims=[self.index_dim], name=name, link_params=False)

        nl = len(kernels)
         
        #build covariance structure
        covariance = [[None for i in range(nl)] for j in range(nl)]
        linked = []
        for i in range(0,nl):
            unique=True
            for j in range(0,nl):
                if i==j or (kernels[i] is kernels[j]):
                    covariance[i][j] = {'c': kernels[i].K, 'ug': kernels[i].update_gradients_full}
                    if i>j:
                        unique=False
                elif cross_covariances.get((i,j)) is not None: #cross covariance is given
                    covariance[i][j] = cross_covariances.get((i,j))
                elif kernels[i].name == 'diffKern' and kernels[i].base_kern == kernels[j]: # one is derivative of other
                    covariance[i][j] = {'c': kernels[i].dK_dX_wrap, 'ug': kernels[i].update_gradients_dK_dX}
                    unique=False
                elif kernels[j].name == 'diffKern' and kernels[j].base_kern == kernels[i]: # one is derivative of other
                    covariance[i][j] = {'c': kernels[j].dK_dX2_wrap, 'ug': kernels[j].update_gradients_dK_dX2}
                elif kernels[i].name == 'diffKern' and kernels[j].name == 'diffKern' and kernels[i].base_kern == kernels[j].base_kern: #both are partial derivatives
                    covariance[i][j] = {'c': partial(kernels[i].K, dimX2=kernels[j].dimension), 'ug': partial(kernels[i].update_gradients_full, dimX2=kernels[j].dimension)}
                    if i>j:
                        unique=False
                else: # zero matrix
                    covariance[i][j] = {'c': lambda x, x2: np.zeros((x.shape[0],x2.shape[0])), 'ug': lambda x, x2: np.zeros((x.shape[0],x2.shape[0]))}       
            if unique is True:
                linked.append(i)
        self.covariance = covariance
        self.link_parameters(*[kernels[i] for i in linked])
        
    @Cache_this(limit=3, ignore_args=())
    def K(self, X ,X2=None):
        if X2 is None:
            X2 = X
        slices = index_to_slices(X[:,self.index_dim])
        slices2 = index_to_slices(X2[:,self.index_dim])
        target =  np.zeros((X.shape[0], X2.shape[0]))
        #for j in range(len(slices2)):
            #for i in range(len(slices)):
                #for l in range(len(slices2[j])):
                    #for k in range( len(slices[i])):
                        #print(i)
                        #print(j)
                        #print(slices[i][k])
                        #print(X[slices[i][k],:].shape)
                
        [[[[ target.__setitem__((slices[i][k],slices2[j][l]), self.covariance[i][j]['c'](X[slices[i][k],:],X2[slices2[j][l],:])) for k in range( len(slices[i]))] for l in range(len(slices2[j])) ] for i in range(len(slices))] for j in range(len(slices2))]  
        return target

    @Cache_this(limit=3, ignore_args=())
    def Kdiag(self,X):
        slices = index_to_slices(X[:,self.index_dim])
        kerns = itertools.repeat(self.kern) if self.single_kern else self.kern
        target = np.zeros(X.shape[0])
        [[np.copyto(target[s], kern.Kdiag(X[s])) for s in slices_i] for kern, slices_i in zip(kerns, slices)]
        return target

    def reset_gradients(self):
        for kern in self.kern: kern.reset_gradients()

    def update_gradients_full(self,dL_dK,X,X2=None, reset=True):
        if reset:
            self.reset_gradients()
        if X2 is None:
            X2 = X
        slices = index_to_slices(X[:,self.index_dim])
        slices2 = index_to_slices(X2[:,self.index_dim])                
        [[[[ self.covariance[i][j]['ug'](dL_dK[slices[i][k],slices2[j][l]], X[slices[i][k],:], X2[slices2[j][l],:], False) for k in range(len(slices[i]))] for l in range(len(slices2[j]))] for i in range(len(slices))] for j in range(len(slices2))]
        #print('new iter')
        #print("dldk: {}".format(np.sum(dL_dK)))
        #print("ls {}".format(self.kern[0].lengthscale.values))
        #print("v {}".format(self.kern[0].variance.values))
        #print("lengthscale gradient: {}".format(self.kern[0].lengthscale.gradient))
        #print("variance gradient: {}".format(self.kern[0].variance.gradient))
        #d = 0.00001
        #k = self.K(X,X2)
        #lg_orig=self.kern[0].lengthscale
        #self.kern[0].lengthscale._update_on = False
        #self.kern[0].lengthscale += d
        #k2 = self.K(X,X2)
        #self.kern[0].lengthscale = lg_orig
        #print("ref lg g")
        #print((k2-k)/d)
        #self.kern[0].lengthscale._update_on = True
        #ko = self.K(X,X2)
        #v_orig=self.kern[0].variance
        #self.kern[0].variance._update_on = False
        #self.kern[0].variance += d
        #k2 = self.K(X,X2)
        #self.kern[0].variance = v_orig
        #print("ref v g")
        #print((k2-ko)/d)
        ###print(np.sum(dL_dK))
        #self.kern[0].variance._update_on = True
        ###print(np.sum(ko-k))
        
    def update_gradients_diag(self, dL_dKdiag, X):
        for kern in self.kerns: kern.reset_gradients()
        slices = index_to_slices(X[:,self.index_dim])
        kerns = itertools.repeat(self.kern) if self.single_kern else self.kern
        [[ self.kerns[i].update_gradients_diag(dL_dKdiag[slices[i][k]], X[slices[i][k],:], False) for k in range(len(slices[i]))] for i in range(len(slices))]
    
    def gradients_X(self,dL_dK, X, X2=None):
        assert 0, "gradients_X"
    
    def gradients_X_diag(self, dL_dKdiag, X):
        assert 0, "gradients_X_diag"