import abc
import numpy as np
cimport numpy as np
from libc.math cimport sin, log, exp
import scipy.special as sp
import scipy.optimize as so

class Model(metaclass=abc.ABCMeta):
    """Base abstract class for posterior model.
    
    All inherited models must implement the abstract constructor, fit and 
    predict methods.
    """

    @abc.abstractmethod
    def __init__(self, int ndim, list domains not None, hyper_parameters=None):
        """Constructor setting the hyper parameters of the model.
        
        Must be implemented by derivied class (currently abstract).
        
        Args: 
            domains: List of 1D numpy ndarrays containing the 
                domains for each parameter of model.  Each domain 
                is of length two, specifying a lower and upper bound for real
                hyper parameters but can be different in other cases if required.  
            hyper_parameters: Hyperparameters for model.
        """

    @abc.abstractmethod
    def fit(self, np.ndarray[double, ndim=2, mode="c"] X, 
            np.ndarray[double, ndim=1, mode="c"] Y):
        """Fit the parameters of the model.
        
        Must be implemented by derivied class (currently abstract).
        
        Args:
            X: 2D array of samples of shape (nsamples, ndim).
            Y: 1D array of target posterior values for each sample in X 
                of shape (nsamples).
        
        Returns:
            Boolean specifying whether fit successful.
        """

    @abc.abstractmethod
    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Predict the value of the posterior at point x.
        
        Must be implemented by derivied class (currently abstract).
        
        Args: 
            x: 1D array of sample of shape (ndim) to predict posterior value.
        
        Return:
            Predicted posterior value.
        """


cdef double HyperSphereObjectiveFunction(double R_squared, X, Y, \
                              centres, inv_covarience):
    """ Evaluates the ojective function with the HyperSphere model
    """

    cdef np.ndarray[double, ndim=2, mode="c"] X_here = X
    cdef np.ndarray[double, ndim=1, mode="c"] Y_here = Y, \
                            centres_here = centres, inv_covarience_here = inv_covarience

    cdef int i_dim, i_sample, ndim = X.shape[1], nsample = X.shape[0]
    cdef double objective = 0.0, distance, mean_shift = np.mean(Y)
    cdef double ln_one_over_volume = ndim*log(R_squared)/2 # Parts that do not depend on R is ignored

    for i_sample in range(nsample):
        distance = 0.0
        for i_dim in range(ndim):
            distance += (X_here[i_sample,i_dim]-centres_here[i_dim])*(X_here[i_sample,i_dim]-centres_here[i_dim])*inv_covarience_here[i_dim]
        if distance < R_squared:
            objective += exp( 2*(mean_shift - Y[i_sample]) ) 

    objective = exp(-2*ln_one_over_volume)*objective/nsample
    # objective = exp(-2*ln_one_over_volume-2*mean_shift)*objective/nsample

    return objective


class HyperSphere(Model):

    def __init__(self, int ndim_in, list domains not None, hyper_parameters=None):
        """ constructor setting the hyper parameters of the model
        """
        if hyper_parameters != None:
            raise ValueError("Hyper Sphere model has no hyper parameters.")
        if len(domains) != 1:
            raise ValueError("Hyper Sphere model domains list should be length 1.")


        self.ndim   = ndim_in

        self.centres_set        = False
        self.centres            = np.zeros((ndim_in))
        self.inv_covarience_set = False
        self.inv_covarience     = np.ones((ndim_in))

        self.R_domain = domains[0]
        self.set_R(np.mean(self.R_domain))

    def set_R(self, double R):
        """ sets the radius of the hyper sphere and calculates the volume of the sphere"""

        if ~np.isfinite(R):
            raise ValueError("Radius is a Nan")
        if R <= 0.0:
            raise ValueError("Radius must be positive")

        self.R                  = R
        self.set_precompucted_values()
        return

    def set_precompucted_values(self):


        cdef int i_dim
        cdef double det_covarience = 1.0

        for i_dim in range(self.ndim):
            det_covarience *= 1.0/self.inv_covarience[i_dim]

        self.R_squared          = self.R*self.R
        self.ln_one_over_volume = -((self.ndim/2)*np.log(np.pi) + self.ndim*np.log(self.R) - sp.gammaln(self.ndim/2+1) + 0.5*log(det_covarience))
        return


    def set_centres(self, np.ndarray[double, ndim=1, mode="c"] centres_in):

        cdef int i_dim

        if centres_in.size != self.ndim:
            raise ValueError("centres size is not equal ndim")

        for i_dim in range(self.ndim):
            if ~np.isfinite(centres_in[i_dim]):
                raise ValueError("Nan/inf's in centres, this may be due to a Nan in the samples")

        for i_dim in range(self.ndim):
            self.centres[i_dim] = centres_in[i_dim]

        self.centres_set = True

        return

    def set_inv_covarience(self, np.ndarray[double, ndim=1, mode="c"] inv_covarience_in):

        cdef int i_dim

        if inv_covarience_in.size != self.ndim:
            raise ValueError("inv_covarience size is not equal ndim")

        for i_dim in range(self.ndim):
            if ~np.isfinite(inv_covarience_in[i_dim]):
                raise ValueError("Nan/inf's in inv_covarience, this may be due to a Nan in the samples")

        for i_dim in range(self.ndim):
            self.inv_covarience[i_dim] = inv_covarience_in[i_dim]

        self.inv_covarience_set = True

        self.set_precompucted_values()

        return

    def fit(self, np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=1, mode="c"] Y):
        """Fit the parameters of the model

        Raises:
            ValueError if the first dimension of X is not the same as Y
            ValueError if the second dimension of X is not the same as ndim
        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y sizes are not the same")

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim")

        if ~self.centres_set:
            self.set_centres(np.mean(X,axis=0))

        if ~self.inv_covarience_set:
            self.set_inv_covarience(np.std(X,axis=0)**(-2))

        result = so.minimize_scalar(HyperSphereObjectiveFunction, bounds=[self.R_domain[0],self.R_domain[1]], \
                           args=(X, Y, self.centres, self.inv_covarience), method='Bounded')

        self.set_R(result.x)

        return result.success

    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the hight of the log_e posterior at point x
        """

        distance = np.dot(x,self.inv_covarience*x)

        if distance < self.R_squared:
            return self.ln_one_over_volume
        else:
            return -np.inf
    
cdef dict set_grid(grid_in, X_in, start_end_in, inv_scales_in, ngid_in):

    cdef dict grid = grid_in
    cdef np.ndarray[double, ndim=2, mode="c"] X = X_in, start_end = start_end_in
    cdef np.ndarray[double, ndim=1, mode="c"] inv_scales = inv_scales_in

    cdef long i_sample, i_dim, sub_index, index, nsamples = X.shape[0], ndim = X.shape[1], ngrid = ngid_in

    for i_sample in range(nsamples):
        index = 0
        for i_dim in range(ndim):
            sub_index = <long>((X_in[i_sample,i_dim]-start_end[0,i_dim])/inv_scales[i_dim]) + 1
            if sub_index < 0 or sub_index >= ngrid:
                pass
            # index += 

#     low_x  = -25.
#     high_x = 25.
#     scale_diag = scale*2
#     index_dict = {}
#     #loop over samples
#     for i_walker in range(n_train):
#         for i_sample in range(samples_per_walker_net_train):
#             #find which pixel it is in
#             i_pixel = int((samples_train[i_walker,i_sample,0]-low_x)/scale_diag)
#             j_pixel = int((samples_train[i_walker,i_sample,1]-low_x)/scale_diag)
#             pixel_key = str(i_pixel)+" "+str(j_pixel)
#             # add its index to list in that pixel
#             if pixel_key in index_dict:
#                 index_dict[pixel_key].append((i_walker,i_sample))
#             else:
#                 index_dict[pixel_key] = [(i_walker,i_sample)]



class KernelDensityEstimate(Model):

    def __init__(self, int ndim, list domains not None, hyper_parameters=None):
        """ constructor setting the hyper parameters of the model
        """
        if len(hyper_parameters) != 1:
            raise ValueError("Kernel Density Estimate hyper parameter list should be length 1.")
        if len(domains) != 0:
            raise ValueError("Kernel Density Estimate domains list should be length 0.")

        self.ndim     = ndim
        self.R        = hyper_parameters[0]

        self.scales_set = False
        self.start_end  = np.zeros((ndim,2))
        self.inv_scales = np.ones((ndim))
        self.distance   = self.R*self.R/4
        self.ngrid      = <int>(1.0/self.R+1E-8)+2

        self.grid       = {}

        return

    def set_scales(self, np.ndarray[double, ndim=2, mode="c"] X):

        cdef int i_dim

        for i_dim in range(self.ndim):
            self.inv_scales[i_dim]  = 1.0/(self.R * (X[:,i_dim].max() - X[:,i_dim].min()))
            self.start_end[i_dim,0] = X[:,i_dim].min()
            self.start_end[i_dim,1] = X[:,i_dim].max()

        self.scales_set = True

        return

    def fit(self, np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=1, mode="c"] Y):
        """Fit the parameters of the model

        Raises:
            ValueError if the first dimension of X is not the same as Y
            ValueError if the second dimension of X is not the same as ndim
        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y sizes are not the same")

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim")

        self.set_scales(X)

        set_grid(self.grid, X, self.start_end, self.inv_scales, self.ngrid)

        #set dictionary 

        return

    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the hight of the posterior at point x
        """
        # similar to this
# def posterior_model_dict(vec_x, samples_train, index_dict, low_x, radius):
#     #find out the pixel that the sample is in
#     i_pixel_org = int((vec_x[0]-low_x)/(radius*2))
#     j_pixel_org = int((vec_x[1]-low_x)/(radius*2))
#     norm_circle = 1.0/(samples_train.shape[0]*samples_train.shape[1]*np.pi*radius**2)
#     model_value = 0.0

#     for i_pixel in range(i_pixel_org-1,i_pixel_org+2):
#         for j_pixel in range(j_pixel_org-1,j_pixel_org+2):
#             pixel_key = str(i_pixel)+" "+str(j_pixel)
#             # print (vec_x[0]-low_x)/(scale*2), i_pixel
#             # loop over list in that pixel
#             if pixel_key in index_dict:
#                 for sample_index in index_dict[pixel_key]:
#                     # do what I did before
#                     length = np.dot(vec_x-samples_train[sample_index[0],sample_index[1],:], vec_x-samples_train[sample_index[0],sample_index[1],:])
#                     if length<radius**2:
#                         model_value += norm_circle

#     return model_value

#     return model_value

        return


class ModifiedGaussianMixtureModel(Model):

    def __init__(self, int ndim, list domains not None, hyper_parameters=None):
        """ constructor setting the hyper parameters of the model
        """

    def fit(self, np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=1, mode="c"] Y):
        """Fit the parameters of the model
        """
        return

    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the hight of the posterior at point x
        """
        return
