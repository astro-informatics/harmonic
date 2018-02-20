import abc
import numpy as np
cimport numpy as np
from libc.math cimport sin, log, exp
import scipy.special as sp
import scipy.optimize as so

class Model(metaclass=abc.ABCMeta):
    """Base class for model
    
    All inherited models must implement the abstract constructor and fit and predict methods.
    """

    @abc.abstractmethod
    def __init__(self, int ndim, list domains not None, hyper_parameters=None):
        """Constructor setting the hyper parameters of the model
        
        Must be implemented by derivied class (currently abstract).
        
        Args: 
            domains: List of 1D numpy ndarrays containing the 
                domains for each parameter of model.  Each domain 
                is of length two, specifying a lower and upper bound for real
                hyper parameters but can be different in other cases if required.  
            hyper_parameters: Hyperparameters for model.
        """

    @abc.abstractmethod
    def fit(self, np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=1, mode="c"] Y):
        """Fit the parameters of the model        
        
        Must be implemented by derivied class (currently abstract).
        
        Args:
            X: 2D array of samples of shape nsamples x ndim.
            Y: 1D array of target posterior values for each sample in X (shape nsamples).
        
        Returns:
            Boolean specifying whether fit successful.
        """

    @abc.abstractmethod
    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the value of the posterior at point x
        
        Must be implemented by derivied class (currently abstract).
        
        Args: 
            x: 1D array of sample to predict posterior value.
        
        Return:
            Predicted posterior value.
        """

cdef double ObjectiveFunction(double R_squared, X, Y, \
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

        result = so.minimize_scalar(ObjectiveFunction, bounds=[self.R_domain[0],self.R_domain[1]], \
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
    


class KernelDensityEstimate(Model):

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
