import abc
import numpy as np
cimport numpy as np
from libc.math cimport log, exp
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
                              centre, inv_covariance):
    """ Evaluates the ojective function with the HyperSphere model
    """

    cdef np.ndarray[double, ndim=2, mode="c"] X_here = X
    cdef np.ndarray[double, ndim=1, mode="c"] Y_here = Y, \
        centre_here = centre, \
        inv_covariance_here = inv_covariance

    cdef int i_dim, i_sample, ndim = X.shape[1], nsample = X.shape[0]
    cdef double objective = 0.0, distance, mean_shift = np.mean(Y)
    cdef double ln_one_over_volume = ndim*log(R_squared)/2 # Parts that do not depend on R is ignored

    for i_sample in range(nsample):
        distance_squared = 0.0
        for i_dim in range(ndim):
            distance_squared += \
                (X_here[i_sample,i_dim] - centre_here[i_dim]) \
                * (X_here[i_sample,i_dim] - centre_here[i_dim]) \
                * inv_covariance_here[i_dim]
        if distance_squared < R_squared:
            objective += exp( 2*(mean_shift - Y[i_sample]) ) 

    objective = exp(-2*ln_one_over_volume)*objective/nsample
    # objective = exp(-2*ln_one_over_volume-2*mean_shift)*objective/nsample

    return objective


class HyperSphere(Model):

    def __init__(self, int ndim_in, list domains not None, hyper_parameters=None):
        """ constructor setting the hyper parameters of the model
        """
        if hyper_parameters != None:
            raise ValueError("HyperSphere model has no hyper parameters.")
        if len(domains) != 1:
            raise ValueError("HyperSphere model domains list should " +
                "be length 1.")

        self.ndim               = ndim_in
        self.centre_set         = False
        self.centre             = np.zeros((ndim_in))
        self.inv_covariance_set = False
        self.inv_covariance     = np.ones((ndim_in))
        self.R_domain           = domains[0]
        self.set_R(np.mean(self.R_domain))
        self.fitted             = False

    def set_R(self, double R):
        """ sets the radius of the hyper sphere and calculates the volume of the sphere"""

        if ~np.isfinite(R):
            raise ValueError("Radius is a NaN.")
        if R <= 0.0:
            raise ValueError("Radius must be positive.")

        self.R = R
        self.set_precompucted_values()
        
        return

    def set_precompucted_values(self):

        cdef int i_dim
        cdef double det_covariance = 1.0

        for i_dim in range(self.ndim):
            det_covariance *= 1.0/self.inv_covariance[i_dim]

        self.R_squared = self.R*self.R
        
        # Compute log_e(1/volume).        
        # First compute volume of hypersphere then adjust for transformation 
        # by C^{1/2} to give hyper-ellipse by multiplying by 0.5*det(C).        
        volume_hypersphere = (self.ndim/2)*log(np.pi) \
            + self.ndim*log(self.R) - sp.gammaln(self.ndim/2+1)
        self.ln_one_over_volume = \
            - volume_hypersphere - 0.5*log(det_covariance) 
            
        return

    def set_centre(self, np.ndarray[double, ndim=1, mode="c"] centre_in):

        cdef int i_dim

        if centre_in.size != self.ndim:
            raise ValueError("centre size is not equal ndim.")

        for i_dim in range(self.ndim):
            if ~np.isfinite(centre_in[i_dim]):
                raise ValueError("NaN/Inf's in inv_covariance (may be due " + 
                                 "to a NaN in samples).")

        for i_dim in range(self.ndim):
            self.centre[i_dim] = centre_in[i_dim]

        self.centre_set = True

        return

    def set_inv_covariance(self, np.ndarray[double, ndim=1, mode="c"] 
                           inv_covariance_in):

        cdef int i_dim

        if inv_covariance_in.size != self.ndim:
            raise ValueError("inv_covariance size is not equal ndim.")

        for i_dim in range(self.ndim):
            if ~np.isfinite(inv_covariance_in[i_dim]):
                raise ValueError("NaN/Inf's in inv_covariance (may be due " + 
                                 "to a NaN in samples).")
            if inv_covariance_in[i_dim] <= 0.0:
                raise ValueError("Inverse Covariance values must " +
                                "be positive.")

        for i_dim in range(self.ndim):
            self.inv_covariance[i_dim] = inv_covariance_in[i_dim]

        self.inv_covariance_set = True

        self.set_precompucted_values()

        return

    def fit(self, np.ndarray[double, ndim=2, mode="c"] X, 
            np.ndarray[double, ndim=1, mode="c"] Y):
        """Fit the parameters of the model

        Raises:
            ValueError if the first dimension of X is not the same as Y
            ValueError if the second dimension of X is not the same as ndim
        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y sizes are not the same.")

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim.")

        if ~self.centre_set:
            self.set_centre(np.mean(X, axis=0))

        if ~self.inv_covariance_set:
            self.set_inv_covariance(np.std(X, axis=0)**(-2))

        result = so.minimize_scalar(HyperSphereObjectiveFunction, 
            bounds=[self.R_domain[0], self.R_domain[1]], 
            args=(X, Y, self.centre, self.inv_covariance), 
            method='Bounded')

        self.set_R(result.x)

        self.fitted = result.success

        return result.success

    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the hight of the log_e posterior at point x
        """

        x_minus_centre = x - self.centre        
        
        distance_squared = \
            np.dot(x_minus_centre, x_minus_centre*self.inv_covariance)

        if distance_squared < self.R_squared:
            return self.ln_one_over_volume
        else:
            return -np.inf
    
cdef dict set_grid(grid_in, X_in, start_end_in, inv_scales_in, ngid_in, D_in):

    cdef dict grid = grid_in
    cdef np.ndarray[double, ndim=2, mode="c"] X = X_in, start_end = start_end_in
    cdef np.ndarray[double, ndim=1, mode="c"] inv_scales = inv_scales_in

    cdef long i_sample, i_dim, sub_index, index, nsamples = X.shape[0], ndim = X.shape[1], ngrid = ngid_in
    cdef double inv_diam = 1.0/D_in

    for i_sample in range(nsamples):
        index = 0
        for i_dim in range(ndim):
            sub_index = <long>((X_in[i_sample,i_dim]-start_end[i_dim,0])*inv_scales[i_dim]*inv_diam) + 1
            index += sub_index*ngrid**i_dim
            # print(i_dim, X_in[i_sample,i_dim], start_end[i_dim,0], inv_scales[i_dim], (X_in[i_sample,i_dim]-start_end[i_dim,0])*inv_scales[i_dim], sub_index, ngrid, index)
        if index in grid:
            grid[index].append(i_sample)
        else:
            grid[index] = [i_sample]


class KernelDensityEstimate(Model):

    def __init__(self, int ndim, list domains not None, hyper_parameters=None):
        """ constructor setting the hyper parameters of the model
        """
        if len(hyper_parameters) != 1:
            raise ValueError("Kernel Density Estimate hyper parameter list should be length 1.")
        if len(domains) != 0:
            raise ValueError("Kernel Density Estimate domains list should be length 0.")

        self.ndim     = ndim
        self.D        = hyper_parameters[0]

        self.scales_set         = False
        self.start_end          = np.zeros((ndim,2))
        self.inv_scales         = np.ones((ndim))
        self.inv_scales_squared = np.ones((ndim))
        self.distance           = self.D*self.D/4
        self.ngrid              = <int>(1.0/self.D+1E-8)+3
        self.ln_norm            = 0.0
        self.fitted             = False

        self.grid       = {}

        return

    def set_scales(self, np.ndarray[double, ndim=2, mode="c"] X):

        cdef int i_dim

        for i_dim in range(self.ndim):
            self.inv_scales[i_dim]  = 1.0/(X[:,i_dim].max() - X[:,i_dim].min())
            self.start_end[i_dim,0] = X[:,i_dim].min()
            self.start_end[i_dim,1] = X[:,i_dim].max()

        self.inv_scales_squared = self.inv_scales**2

        self.scales_set = True

        return

    def precompute_normalising_factor(self, np.ndarray[double, ndim=2, mode="c"] X):
        cdef double ln_volume, det_covarience = 1.0

        for i_dim in range(self.ndim):
            det_covarience *= 1.0/self.inv_scales[i_dim]

        ln_volume = ((self.ndim/2)*log(np.pi) + self.ndim*log(self.D/2) - sp.gammaln(self.ndim/2+1) + log(det_covarience))
         

        self.ln_norm = log(<double>X.shape[0]) + ln_volume
        pass



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

        self.samples = X.copy() # TODO consider functionality for shallow copy to save mem

        self.set_scales(self.samples)

        #set dictionary 
        set_grid(self.grid, self.samples, self.start_end, self.inv_scales, self.ngrid, self.D)

        self.precompute_normalising_factor(self.samples)

        self.fitted = True

        return True

    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the hight of the posterior at point x
        """
        # similar to this
def posterior_model_dict(vec_x, samples_train, index_dict, low_x, radius):
    #find out the pixel that the sample is in
    i_pixel_org = int((vec_x[0]-low_x)/(radius*2))
    j_pixel_org = int((vec_x[1]-low_x)/(radius*2))
    norm_circle = 1.0/(samples_train.shape[0]*samples_train.shape[1]*np.pi*radius**2)
    model_value = 0.0

    for i_pixel in range(i_pixel_org-1,i_pixel_org+2):
        for j_pixel in range(j_pixel_org-1,j_pixel_org+2):
            pixel_key = str(i_pixel)+" "+str(j_pixel)
            # print (vec_x[0]-low_x)/(scale*2), i_pixel
            # loop over list in that pixel
            if pixel_key in index_dict:
                for sample_index in index_dict[pixel_key]:
                    # do what I did before
                    length = np.dot(vec_x-samples_train[sample_index[0],sample_index[1],:], vec_x-samples_train[sample_index[0],sample_index[1],:])
                    if length<radius**2:
                        model_value += norm_circle

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
