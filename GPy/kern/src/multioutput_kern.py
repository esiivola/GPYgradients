from .kern import Kern
import numpy as np
from functools import reduce

class MultioutputKern(Kern):
    def __init__(self, kern_list, cross_covariances={}):
        #assert type(kern_list) is IntType, "Kernel list is not a list"
        #assert type(cross_covariance) is IntType, "Kernel list is not a list"
        nl = len(kern_list)
        #self.input_dim = [None]*nl
        #self.active_dims = [None]*nl
        #self.useGPU = False
        #for i in range(0,nl):
            #kern = kern_list[i]
            #self.input_dim[i] = kern.input_dim
            #self.active_dims[i] = kern.active_dims
            #self.useGPU = True if kern.useGPU else False
        active_dims = reduce(np.union1d, (np.r_[x.active_dims] for x in kern_list))
        input_dim = active_dims.size
        # initialize the kernel with the full input_dim
        super(MultioutputKern, self).__init__(input_dim, active_dims, 'MultioutputKern')
         
        #build covariance structure
        covariance = [[None]*nl]*nl
        linked = []
        for i in range(0,nl):
            unique=True
            for j in range(0,nl):
                if i==j:
                    covariance[i][i] = {'c': kern_list[j].K, 'ug': lambda dl_dk, x, x2: kern_list[i].update_gradients_full(dl_dk, x, x2, False)}
                elif cross_covariances.get((i,j)) is not None: #cross covariance is given
                    covariance[i][j] = cross_covariances.get((i,j))
                elif kern_list[i].name == 'diffKern' and kern_list[i].base_kern == kern_list[j]: # one is derivative of other
                    covariance[i][j] = {'c': kern_list[i].dK_dX, 'ug': lambda dl_dk, x, x2: kern_list[i].update_gradients_dK_dX(dl_dk, x, x2, False)}
                    unique=False
                elif kern_list[j].name == 'diffKern' and kern_list[j].base_kern == kern_list[i]: # one is derivative of other
                    covariance[i][j] = {'c': lambda x,x2: np.kern_list[j].dK_dX2(x, x2), 'ug': lambda dk_dl, x, x2: kern_list[j].update_gradients_dK_dX2(dk_dl, x, x2, False)}
                elif kern_list[i].name == 'diffKern' and kern_list[j].name == 'diffKern' and kern_list[i].base_kern == kern_list[j].base_kern: #both are partial derivatives
                    covariance[i][j] = {'c': lambda x, x2: kern_list[i].K(x, x2, kern_list[j].dimension), 'ug': lambda dk_dl, x, x2: kern_list[i].update_gradients_full(dk_dl, x, x2, kern_list[j].dimension, False)}
                    if i>j:
                      unique=False  
                else: # zero matrix
                    covariance[i][j] = {'c': lambda x, x2: np.zeros(x.shape[0],x2.shape[0]), 'ug': lambda x, x2: np.zeros(x.shape[0],x2.shape[0])}       
            if unique is True:
                linked.append(i)
        self.kern_list = kern_list
        self.covariance = covariance
        self.link_parameters(*[kern_list[i] for i in linked])

    def parameters_changed(self):
        for kern in self.kern_list: kern.parameters_changed()

    def K(self, X_list, X2_list): # NOTE: If any of the elements in  X or X2 are None, they are treated as non existent
        x_i = [i for i, e in enumerate(X_list) if e is not None]
        x2_i = [i for i, e in enumerate(X2_list) if e is not None]
        k = None
        for i in x_i:
            k_temp = None
            for j in x2_i:
                if j <= i:
                    t = self.covariance[i][j]['c'](X_list[i], X2_list[j])
                else:
                    t = np.transpose(self.covariance[j][i]['c'](X2_list[j], X_list[i]))
                k_temp = t if k_temp is None else np.bmat([k_temp, t])
            k = k if k_temp is None else np.bmat([[k], [k_temp]])
 
    def Kdiag(self, X_list):
        x_i = [i for i, e in enumerate(X_list) if e is not None]
        k = None
        for i in x_i:
            ind_t = Xi_list[i]
            t = self.kern_list[i].Kdiag(X[i])
            k = k if k_temp is None else np.r_[k, t]
        return k
    

    def update_gradients_diag(self, dL_dKdiag, X_list):
        """ update the gradients of all parameters when using only the diagonal elements of the covariance matrix"""
        for kern in kern_list: kern.reset_gradients()
        nl = len(X_list)
        ind_list = [0]*(nl+1)
        for i in range(0,nl):
            ind_list[i+1] = ind_list[i] + X_list[i].shape[0]
        x_i = [i for i, e in enumerate(X_list) if e is not None]
        for i in x_i:
            kern_list[i].update_gradients_diag(dL_dKdiag[ind_list[i]:ind_list[i+1]], X_list[i], False)
        

    def update_gradients_full(self, dL_dK, X_list, X2_list):
        """Set the gradients of all parameters when doing full (N) inference."""
        for kern in kern_list: kern.reset_gradients()
        x_i = [i for i, e in enumerate(X_list) if e is not None]
        x2_i = [i for i, e in enumerate(X2_list) if e is not None]
        nl = len(X_list)
        ind_list = [0]*(nl+1)
        ind_list2 = [0]*(nl+1)
        for i in range(0,nl):
            ind_list[i+1] = ind_list[i] + X_list[i].shape[0]
            ind_list2[i+1] = ind_list2[i] + X2_list[i].shape[0]
        for i in x_i:
            for j in x2_i:
                if j <= i:
                    self.covariance[i][j]['ug'](dL_dK[ind_list[i]:ind_list[i+1],ind_list2[j]:ind_list2[j+1]], X_list[i], X2_list[j])
                else:
                    self.covariance[j][i]['ug'](np.transpose(dL_dK[ind_list[i]:ind_list[i+1],ind_list2[j]:ind_list2[j+1]]), X2_list[j], X_list[i])
                    
    @property
    def parts(self):
        return self.parameters

    def _set_all_dims_ative(self):
        self._all_dims_active = np.atleast_1d(self.active_dims).astype(int)

    def input_sensitivity(self, summarize=True):
        """
        If summize is true, we want to get the summerized view of the sensitivities,
        otherwise put everything into an array with shape (#kernels, input_dim)
        in the order of appearance of the kernels in the parameterized object.
        """
        if not summarize:
            num_params = [0]
            parts = []
            def sum_params(x):
                if (not isinstance(x, CombinationKernel)) and isinstance(x, Kern):
                    num_params[0] += 1
                    parts.append(x)
            self.traverse(sum_params)
            i_s = np.zeros((num_params[0], self.input_dim))
            from operator import setitem
            [setitem(i_s, (i, k._all_dims_active), k.input_sensitivity(summarize)) for i, k in enumerate(parts)]
            return i_s
        else:
            raise NotImplementedError("Choose the kernel you want to get the sensitivity for. You need to override the default behaviour for getting the input sensitivity to be able to get the input sensitivity. For sum kernel it is the sum of all sensitivities, TODO: product kernel? Other kernels?, also TODO: shall we return all the sensitivities here in the combination kernel? So we can combine them however we want? This could lead to just plot all the sensitivities here...")

    def _check_active_dims(self, X):
        return

    def _check_input_dim(self, X):
        # As combination kernels cannot always know, what their inner kernels have as input dims, the check will be done inside them, respectively
        return