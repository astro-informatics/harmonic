import abc
import numpy as np
cimport numpy as np
from libc.math cimport log, exp, sqrt, M_PI
import scipy.special as sp
import scipy.optimize as so
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Model(metaclass=abc.ABCMeta):
    """Base abstract class for posterior model.
    
    All inherited models must implement the abstract constructor, fit and 
    predict methods.
    """

    @abc.abstractmethod
    def __init__(self, long ndim, list domains not None, hyper_parameters=None):
        """Constructor setting the hyper parameters and domains of the model.
        
        Must be implemented by derivied class (currently abstract).
        
        Args: 
            long ndim: Dimension of the problem to solve.
            list domains: List of 1D numpy ndarrays containing the 
                domains for each parameter of model.  Each domain 
                is of length two, specifying a lower and upper bound for real
                hyper parameters but can be different in other cases if required.  
            list hyper_parameters: Hyperparameters for model.
        """

    @abc.abstractmethod
    def fit(self, np.ndarray[double, ndim=2, mode="c"] X, 
            np.ndarray[double, ndim=1, mode="c"] Y):
        """Fit the parameters of the model.
        
        Must be implemented by derivied class (currently abstract).
        
        Args:
            X: 2D array of samples of shape (nsamples, ndim).
            Y: 1D array of target log_e posterior values for each sample in X 
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
            Predicted log_e posterior value.
        """

    @abc.abstractmethod
    def is_fitted(self):
        """Specify whether model has been fitted.
        
        Args: 
            None.
            
        Return:
            Boolean specifying whether the model has been fitted.
        """


#===============================================================================
# Hyper sphere model 
#===============================================================================

cdef double HyperSphereObjectiveFunction(double R_squared, X, Y, \
                                         centre, inv_covariance, mean_shift):
    """Evaluate ojective function forthe HyperSphere model. Objective function
    is given by the variance of the estimator (subject to a linear
    transformation that does not depend on the radius of the sphere, which is
    the variable to be fitted).

    Args:
        double R_squared: Radius of the hyper sphere squared.
        X: 2D numpy.ndarray containing the samples 
            with shape (nsamples, ndim) and dtype double.
        Y: 1D numpy.ndarray containing the log_e posterior 
            values with shape (nsamples) and dtype double.
        centre: 1D numpy.ndarray containing the centre of the sphere 
            with shape (ndim) and dtype double.            
        inv_covariance_in: 1D numpy.ndarray containing the diagonal of  inverse
            covariance matrix that defines the ellipse with shape (ndim) and 
            dtype double.            

    Return:
        Value of the objective function.
    """

    cdef np.ndarray[double, ndim=2, mode="c"] X_here = X
    cdef np.ndarray[double, ndim=1, mode="c"] Y_here = Y, \
        centre_here = centre, \
        inv_covariance_here = inv_covariance

    cdef long i_dim, i_sample, ndim = X.shape[1], nsample = X.shape[0]
    cdef double objective = 0.0, distance, mean_shift_here = mean_shift
    cdef double ln_volume = ndim*log(R_squared)/2  # Parts that do not depend
                                                   # on R are ignored.

    for i_sample in range(nsample):
        distance_squared = 0.0
        for i_dim in range(ndim):
            distance_squared += \
                (X_here[i_sample,i_dim] - centre_here[i_dim]) \
                * (X_here[i_sample,i_dim] - centre_here[i_dim]) \
                * inv_covariance_here[i_dim]
        if distance_squared < R_squared:
            objective += exp( 2*(mean_shift_here - Y[i_sample]) ) 

    objective = exp(-2*ln_volume)*objective/nsample
    
    # If were to correct for mean shift do the following (however, not 
    # necessary when minimising objective function).
    # objective = exp(-2*ln_volume-2*mean_shift_here)*objective/nsample

    return objective


class HyperSphere(Model):
    """HyperSphere Model to approximate the log_e posterior by a 
    hyper-ellipsoid.
    """

    def __init__(self, long ndim_in, list domains not None, hyper_parameters=None):
        """Constructor setting the parameters of the model.

        Args:
            long ndim_in: Dimension of the problem to solve.
            list domains: A list of length 1 containing a 1D array
                of length 2 containing the lower and upper bound of the
                radius of the hyper-sphere.
            hyper_parameters: Should not be set as there are no hyperparameters
                for this model (in general, however, models can have hyperparameters).
                
        Returns: 
            None

        Raises:
            ValueError: If the hyper_parameters variable is not None.
            ValueError: If the length of domains list is not one.
            ValueError: If the ndim_in is not positive.
        """
        
        if hyper_parameters != None:
            raise ValueError("HyperSphere model has no hyperparameters.")
        if len(domains) != 1:
            raise ValueError("HyperSphere model domains list should " +
                "be length 1.")
        if ndim_in < 1:
            raise ValueError("Dimension must be greater than 0.")

        self.ndim               = ndim_in
        self.centre_set         = False
        self.centre             = np.zeros((ndim_in))
        self.inv_covariance_set = False
        self.inv_covariance     = np.ones((ndim_in))
        self.R_domain           = domains[0]
        self.set_R(np.mean(self.R_domain))
        self.fitted             = False

    def is_fitted(self):
        """Specify whether model has been fitted.
        
        Args: 
            None.
            
        Return:
            Boolean specifying whether the model has been fitted.
        """

        return self.fitted

    def set_R(self, double R):
        """Set the radius of the hypersphere and calculate its volume.

        Args:
            double R: The radius of the hyper-sphere.

        Returns:
            None
        
        Raises:
            ValueError: If the radius is a NaN.
            ValueError: If the Raises is not positive.
        """

        if not np.isfinite(R):
            raise ValueError("Radius is a NaN.")
        if R <= 0.0:
            raise ValueError("Radius must be positive.")

        self.R = R
        self.set_precompucted_values()
        
        return

    def set_precompucted_values(self):
        """Precompute volume of the hyper sphere (scaled ellipse) and squared 
        radius.

        Args:
            None

        Raises:
            None

        Returns:
            None
        """
        
        cdef long i_dim
        cdef double det_covariance = 1.0

        for i_dim in range(self.ndim):
            det_covariance *= 1.0/self.inv_covariance[i_dim]

        self.R_squared = self.R*self.R
        
        # Compute log_e(1/volume).        
        # First compute volume of hypersphere then adjust for transformation 
        # by C^{1/2} to give hyper-ellipse by multiplying by det(C)^0.5.        
        volume_hypersphere = (self.ndim/2)*log(np.pi) \
            + self.ndim*log(self.R) - sp.gammaln(self.ndim/2+1)
        self.ln_one_over_volume = \
            - volume_hypersphere - 0.5*log(det_covariance) 
            
        return

    def set_centre(self, np.ndarray[double, ndim=1, mode="c"] centre_in):
        """Set centre of the hyper-sphere.

        Args:
            centre_in: 1D numpy.ndarray containing the centre of sphere 
                with shape (ndim) and dtype double.

        Returns:
            None
            
        Raises:
            ValueError: If the length of the centre array is not the same as
                ndim
            ValueError: If the centre array contains a NaN
        """

        cdef long i_dim

        if centre_in.size != self.ndim:
            raise ValueError("centre size is not equal ndim.")

        for i_dim in range(self.ndim):
            if not np.isfinite(centre_in[i_dim]):
                raise ValueError("NaN/Inf's in inv_covariance (may be due " + 
                                 "to a NaN in samples).")

        for i_dim in range(self.ndim):
            self.centre[i_dim] = centre_in[i_dim]

        self.centre_set = True

        return

    def set_inv_covariance(self, np.ndarray[double, ndim=1, mode="c"] 
                           inv_covariance_in):
        """Set diagonal inverse covariances for the hyper-sphere.
        
        Only diagonal covariance structure is supported.

        Args:
            inv_covariance_in: 1D numpy.ndarray containing the diagonal of 
                inverse covariance matrix that defines the ellipse
                with shape (ndim) and dtype double.
                
        Returns:
            None

        Raises:
            ValueError: If the length of the inv_covariance array is not equal 
                to ndim.
            ValueError: If the inv_covariance array contains a NaN.
            ValueError: If the inv_covariance array contains a value that is 
                not positive.
        """

        cdef long i_dim

        if inv_covariance_in.size != self.ndim:
            raise ValueError("inv_covariance size is not equal ndim.")

        for i_dim in range(self.ndim):
            if not np.isfinite(inv_covariance_in[i_dim]):
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
        """Fit the parameters of the model (i.e. its radius).

        Args:
            X: 2D array of samples of shape (nsamples, ndim).
            Y: 1D array of target log_e posterior values for each sample in X 
                of shape (nsamples).
        
        Returns:
            Boolean specifying whether fit successful.

        Raises:
            ValueError if the first dimension of X is not the same as Y.
            ValueError if the second dimension of X is not the same as ndim.
        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y sizes are not the same.")

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim.")

        if not self.centre_set:
            self.set_centre(np.mean(X, axis=0))

        if not self.inv_covariance_set:
            self.set_inv_covariance(np.std(X, axis=0)**(-2))

        mean_shift = np.mean(Y)
        result = so.minimize_scalar(HyperSphereObjectiveFunction, 
            bounds=[self.R_domain[0], self.R_domain[1]], 
            args=(X, Y, self.centre, self.inv_covariance, mean_shift), 
            method='Bounded')

        self.set_R(sqrt(result.x))

        self.fitted = result.success

        return result.success

    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the value of log_e posterior at point x.

        Args: 
            x: 1D array of sample of shape (ndim) to predict posterior value.
        
        Return:
            Predicted posterior value.
            
        Raises:
            None
        """
        
        x_minus_centre = x - self.centre        
        
        distance_squared = \
            np.dot(x_minus_centre, x_minus_centre*self.inv_covariance)

        if distance_squared < self.R_squared:
            return self.ln_one_over_volume
        else:
            return -np.inf


#===============================================================================
# Kernel Density Estimation model
#===============================================================================

cdef KernelDensityEstimate_set_grid(dict grid, \
                                    np.ndarray[double, ndim=2, mode="c"] X, 
                                    np.ndarray[double, ndim=2, mode="c"] start_end, \
                                    np.ndarray[double, ndim=1, mode="c"] inv_scales, \
                                    long ngrid, double D):    
    """Creates a dictionary that allows a fast way to find the indexes of
    samples in a pixel in a grid where the pixel sizes are the diameter of the
    hyper spheres placed at each sample.

    Args:        
        dict grid: Empty dictionary where the list of the sample index will be
            placed. The key is an index of the grid (c type ordering) and the 
            value is a list containing the indexes in the sample array of all 
            the samples in that index.            
        X: 2D array of samples of shape (nsamples, ndim).
        Y: 1D array of target log_e posterior values for each sample in X 
            of shape (nsamples).
        start_end:  2D array of the lowest and highest sample in each dimension 
            (ndim,2).        
        inv_scales: 1D array of the 1.0/delta_x_i where delta_x_i is the
            difference between the max and min of the sample in dimension 
            i.        
        long ngrid: Number of pixels in each dimension in the grid.
        double D: Diameter of the hyper sphere.

    Returns:
        None
    """

    cdef long i_sample, i_dim, sub_index, index, nsamples = X.shape[0], ndim = X.shape[1]
    cdef double inv_diam = 1.0/D

    for i_sample in range(nsamples):
        index = 0
        for i_dim in range(ndim):
            sub_index = <long>((X[i_sample,i_dim]-start_end[i_dim,0])*inv_scales[i_dim]*inv_diam) + 1
            index += sub_index*ngrid**i_dim
        if index in grid:
            grid[index].append(i_sample)
        else:
            grid[index] = [i_sample]

cdef KernelDensityEstimate_loop_round_and_search(long index, long i_dim, long ngrid,\
                                                 long ndim, dict grid, \
                                                 np.ndarray[double, ndim=2, mode="c"] samples, \
                                                 np.ndarray[double, ndim=1, mode="c"] x, \
                                                 np.ndarray[double, ndim=1, mode="c"] inv_scales, \
                                                 double radius_squared, long *count):    
    """Recursive function that calls itself in order to call the search_in_pixel
    function on one pixel behind and infront of the pixel x is in for each
    dimension.

    Args:
        long index: The current pixel we are looking at.
        long i_dim: Dimension we are doing the current moving forward and   
            backward in.
        long ngrid: Number of pixels in each dimension in the grid.
        long ndim: Dimension of the problem.        
        dict grid: The dictionary with information on which samples are in
            which pixel. The key is an index of the grid (c type ordering) and 
            the value is a list containing the indexes in the sample array of 
            all the samples in that index.            
        samples: 2D array of samples of shape (nsamples, ndim).
        x: 1D array of the position we are evaluating the prediction for.
        inv_scales: 1D array of the 1.0/delta_x_i where delta_x_i is the 
            difference between the max and min of the sample in dimension i.
        double radius_squared: Radius squared of the local hypersphere.
        long * count: a pointer to the count integer that counts how many hyper 
            spheres the postion x falls inside.

    Returns:
        None
    """
    # this does create looping boundry conditions but doesn't matter in searching
    # it will simply slow things down very very slightly 
    # (probably less then dealing with it will!)
    if i_dim >= 0:
        for iter_i_dim in range(-1,2):
            KernelDensityEstimate_loop_round_and_search(index+iter_i_dim*ngrid**(i_dim), i_dim-1, ngrid, ndim, grid, samples, x, inv_scales, radius_squared, count)
    else:
        KernelDensityEstimate_search_in_pixel(index, grid, samples, x, inv_scales, radius_squared, count)

cdef KernelDensityEstimate_search_in_pixel(long index, dict grid, \
                                           np.ndarray[double, ndim=2, mode="c"] samples, \
                                           np.ndarray[double, ndim=1, mode="c"] x, \
                                           np.ndarray[double, ndim=1, mode="c"] inv_scales, \
                                           double radius_squared, long *count):
    """Examines all samples that are in the current pixel and counts how many of
    those position x falls inside

    Args:
        long index: The current pixel we are looking at.
        long ndim: Dimension of the problem.
        dict grid: The dictionary with information on which samples are in
            which pixel. The key is an index of the grid (c type ordering) and 
            the value is a list containing the indexes in the sample array of 
            all the samples in that index.
        samples: 2D array of samples of shape (nsamples, ndim).
        x: 1D array of the position we are evaluating the prediction for
        inv_scales: 1D array of the 1.0/delta_x_i where delta_x_i is the 
            difference between the max and min of the sample in dimension i.
        double radius_squared: Radius squared of the local hypersphere.        
        long * count: a pointer to the count integer that counts how many hyper 
            spheres the postion x falls inside.

    Returns:
        None
    """
    
    cdef long sample_index, i_dim, ndim = x.size
    cdef double length = 0.0, dummy
    
    if index in grid:        
        for sample_index in grid[index]:
            length = 0.0
            for i_dim in range(ndim):
                dummy  = x[i_dim]-samples[sample_index,i_dim]
                dummy *= inv_scales[i_dim]
                dummy *= dummy
                length += dummy
            if length<radius_squared:
                count[0] += 1
    return

class KernelDensityEstimate(Model):
    """KernelDensityEstimate model to approximate the log_e posterior using 
    kernel density estimation."""

    def __init__(self, long ndim, list domains not None, hyper_parameters=[0.1]):
        """Constructor setting the hyperparameters and domains of the 
        model.        

        Args:
            long ndim: Dimension of the problem to solve.
            list domains: List of length 0.
            list hyper_parameters: A list of length 1 containing the diameter 
                in scaled units of the hyper spheres to use in the Kernel 
                Density Estimate.

        Raises:
            ValueError: If the hyper_parameters list is not length 1
            ValueError: If the length of domains list is not 0.
            ValueError: If the ndim_in is not positive.
        """
        if len(hyper_parameters) != 1:
            raise ValueError("Kernel Density Estimate hyper parameter list should be length 1.")
        if len(domains) != 0:
            raise ValueError("Kernel Density Estimate domains list should be length 0.")
        if ndim < 1:
            raise ValueError("ndim must be greater then 0.")

        self.ndim     = ndim
        self.D        = hyper_parameters[0]

        self.scales_set         = False
        self.start_end          = np.zeros((ndim,2))
        self.inv_scales         = np.ones((ndim))
        self.inv_scales_squared = np.ones((ndim))
        self.radius_squared           = self.D*self.D/4
        numerical_stability     = 1E-8
        self.ngrid              = <long>(1.0/self.D+numerical_stability)+3 
                                # +3 for 1 extra cell either side and another 
                                # cell for rounding errors.
        self.ln_norm            = 0.0
        self.fitted             = False

        self.grid       = {}

        return

    def is_fitted(self):
        """Specify whether model has been fitted.
        
        Args: 
            None.
            
        Return:
            Boolean specifying whether the model has been fitted.
        """

        return self.fitted
        
    def set_scales(self, np.ndarray[double, ndim=2, mode="c"] X):
        """Set the scales of the hyper spheres based on the min
        and max sample in each dimension.

        Args:
            X: 2D array of samples of shape (nsamples, ndim).

        Returns:
            None

        Raises:
            ValueError if the second dimension of X is not the same as ndim
        """

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim")

        cdef long i_dim

        for i_dim in range(self.ndim):
            xmax = X[:,i_dim].max()
            xmin = X[:,i_dim].min()
            self.inv_scales[i_dim]  = 1.0/(xmax - xmin)
            self.start_end[i_dim,0] = xmin
            self.start_end[i_dim,1] = xmax

        self.inv_scales_squared = self.inv_scales**2

        self.scales_set = True

        return

    def precompute_normalising_factor(self, 
                                      np.ndarray[double, ndim=2, mode="c"] X):
        """Precompute the log_e normalisation factor of the density estimation.

        Args:
            X: 2D array of samples of shape (nsamples, ndim).

        Raises:
            ValueError: if the second dimension of X is not the same as ndim.
        """

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim")

        cdef double ln_volume, det_scaling = 1.0

        for i_dim in range(self.ndim):
            det_scaling *= 1.0/self.inv_scales[i_dim]

        ln_volume = ((self.ndim/2)*log(np.pi) + self.ndim*log(self.D/2) - sp.gammaln(self.ndim/2+1) + log(det_scaling))
        # Not 0.5*log(det_scaling) since constructed from inv_scales not 
        # inv_scales_squared.

        self.ln_norm = log(<double>X.shape[0]) + ln_volume
        pass

    def fit(self, np.ndarray[double, ndim=2, mode="c"] X, 
            np.ndarray[double, ndim=1, mode="c"] Y):
        """Fit the parameters of the model as follows.
            
            1) Set the scales of the model from the samples.
            
            2) Create the dictionary containing all the information on which
            samples are in which pixel in a grid where each pixel size is the
            same as the diameter of the hyper spheres to be placed on each
            sample. The key is an index of the grid (c type ordering) and the
            value is a list containing the indexes in the sample array of all
            the samples in that index 3.
            
            3) Precompute the normalisation factor.

        Args:
            X: 2D array of samples of shape (nsamples, ndim).
            Y: 1D array of target log_e posterior values for each sample in X 
                of shape (nsamples).
        
        Returns:
            Boolean specifying whether fit successful.

        Raises:
            ValueError if the first dimension of X is not the same as Y
            ValueError if the second dimension of X is not the same as ndim
        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y sizes are not the same")

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim")

        self.samples = X.copy() # TODO consider functionality for shallow copy 
                                # to save mem

        self.set_scales(self.samples)

        # Set dictionary 
        KernelDensityEstimate_set_grid(self.grid, self.samples, self.start_end, 
                                       self.inv_scales, self.ngrid, self.D)

        self.precompute_normalising_factor(self.samples)

        self.fitted = True

        return True

    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the value of the posterior at point x.

        Args: 
            x: 1D array of sample of shape (ndim) to predict posterior value.
        
        Return:
            Predicted posterior value.
        """
        cdef np.ndarray[double, ndim=2, mode="c"] samples = self.samples        
        cdef np.ndarray[double, ndim=2, mode="c"] start_end = self.start_end
        cdef np.ndarray[double, ndim=1, mode="c"] inv_scales = self.inv_scales
        cdef long i_sample, i_dim, sub_index, index
        cdef long nsamples = samples.shape[0], ndim = self.ndim
        cdef long ngrid = self.ngrid
        cdef double inv_diam = 1.0/self.D, radius_squared = self.radius_squared
        cdef dict grid = self.grid

        cdef long count = 0
        
        # Find the pixel that the sample is in
        index = 0
        for i_dim in range(ndim):
            sub_index = <long>((x[i_dim]-start_end[i_dim,0])*inv_scales[i_dim]*inv_diam) + 1
            index += sub_index*ngrid**i_dim

        KernelDensityEstimate_loop_round_and_search(index, i_dim, ngrid, ndim, 
                                                    grid, samples, x, 
                                                    inv_scales, radius_squared, 
                                                    &count)

        return log(count) - self.ln_norm


#===============================================================================
# Modified Gausian mixture model 
#===============================================================================

cdef np.ndarray[double, ndim=1, mode="c"] beta_to_weights(\
    np.ndarray[double,  ndim=1, mode="c"] beta, long nguassians):
    """Calculate the weights from the beta_weights.

    Args:
        ndarray beta: 1D array containing the beta values to be converted
            with shape (nguassians).
        long nguassians: The number of Gaussians in the model.

    Return:
        ndarray weights: 1D array where the weight values will go
            with shape (nguassians)

    Raises:
        None
    """
    
    cdef double norm = 0.0
    cdef i_guas
    cdef np.ndarray[double, ndim=1, mode="c"] weights = np.empty(nguassians)

    for i_guas in range(nguassians):
        norm += exp(beta[i_guas])
    norm = 1.0/norm

    for i_guas in range(nguassians):
        weights[i_guas] = exp(beta[i_guas])*norm
    return weights

def beta_to_weights_wrap(np.ndarray[double, ndim=1, mode="c"] beta, 
        long nguassians):    
    """Wrapper to calculate the weights from the beta_weights.

    Args:
        ndarray beta: 1D array containing the beta values to be converted
            with shape (nguassians).
        long nguassians: The number of Gaussians in the model.

    Return:
        ndarray weights: 1D array where the weight values will go
            with shape (nguassians)

    Raises:
        None.
    """    
    
    return beta_to_weights(beta, nguassians)

cdef double calculate_gaussian_normalisation(double alpha, \
    np.ndarray[double, ndim=1, mode="c"] inv_covariance, long ndim):
    """Calculate the normalisation for evaluate_one_guassian.

    Args:
        double alpha: The scalling parameter of the covariance matrix.
        ndarray inv_covariance: 1D array containing the inverse covariance 
            matrix.
        long ndim: Dimension of the problem.

    Returns:
        double: the normalisation factor.
    """
    
    cdef long i_dim
    cdef double det=1.0

    for i_dim in range(ndim):
        det *= alpha*2*M_PI/inv_covariance[i_dim]

    return 1.0/sqrt(det)

def calculate_gaussian_normalisation_wrap(double alpha, \
    np.ndarray[double, ndim=1, mode="c"] inv_covariance, long ndim):
    """Wrapper to calculate the normalisation for evaluate_one_guassian.

    Args:
        double alpha: The scalling parameter of the covariance matrix.
        ndarray inv_covariance: 1D array containing the inverse covariance 
            matrix.
        long ndim: Dimension of the problem.

    Returns:
        double: the normalisation factor.
    """
    
    return calculate_gaussian_normalisation(alpha, inv_covariance, ndim)

cdef double evaluate_one_guassian(np.ndarray[double, ndim=1, mode="c"] x, \
                           np.ndarray[double, ndim=1, mode="c"] mu, \
                           np.ndarray[double, ndim=1, mode="c"] inv_covariance, \
                           double alpha, double weight, long ndim):
    """Evaluate one Guassian.

    Args:
        ndarray x: Postion where the Gaussian is to be evaluated, with shape
            (ndim).
        ndarray mu: Center of the Gaussian, with shape (ndim).
        ndarray inv_covariance: 1D array containing the inverse covariance 
            matrix.
        double alpha: Scalling parameter of the covariance matrix.
        double weight: Weight applied to that Guassian.
        long ndim: Dimension of the problem.

    Returns:
        double: height of the Guassian.
    """
    
    cdef double y, distance = 0.0, norm = calculate_gaussian_normalisation(alpha, inv_covariance, ndim)
    cdef long i_dim

    for i_dim in range(ndim):
        distance += (x[i_dim] - mu[i_dim]) * (x[i_dim] - mu[i_dim]) * inv_covariance[i_dim]
    distance /= 2.0*alpha

    return exp(-distance)*norm*weight

def evaluate_one_guassian_wrap(np.ndarray[double, ndim=1, mode="c"] x, \
                           np.ndarray[double, ndim=1, mode="c"] mu, \
                           np.ndarray[double, ndim=1, mode="c"] inv_covariance, \
                           double alpha, double weight, long ndim):
    """Wrapper to evaluate one Guassian.

    Args:
        ndarray x: Postion where the Gaussian is to be evaluated, with shape
            (ndim).
        ndarray mu: Center of the Gaussian, with shape (ndim).
        ndarray inv_covariance: 1D array containing the inverse covariance 
            matrix.
        double alpha: Scalling parameter of the covariance matrix.
        double weight: Weight applied to that Guassian.
        long ndim: Dimension of the problem.

    Returns:
        double: height of the Guassian.
    """

    return evaluate_one_guassian(x, mu, inv_covariance, alpha, weight, ndim)

cdef double delta_theta_ij(np.ndarray[double, ndim=1, mode="c"] x, \
                    np.ndarray[double, ndim=1, mode="c"] mu, \
                    np.ndarray[double, ndim=1, mode="c"] inv_covariance, \
                    long ndim):
    """Evaluate delta_theta_ij squared which is part of the gradient of the 
    objective function.

    Args:
        ndarray x: 1D array containing the current sample (ndim).
        ndarray mu: 1D array containing the centre of the guassian with
            shape (ndim).
        ndarray inv_covariance: 1D array containing the inverse covariance
            of the guassian with shape (ndim).
        long ndim: Dimension of the problem.

    Returns:
        double delta_theta_ij: Value of delta_theta_ij squared.
    """
    
    cdef long i_dim
    cdef double distance = 0.0, seperation

    for i_dim in range(ndim):
        seperation = x[i_dim]-mu[i_dim]
        distance += seperation*seperation*inv_covariance[i_dim]
    # print(x, inv_covariance, distance)
    return distance


def delta_theta_ij_wrap(np.ndarray[double, ndim=1, mode="c"] x, \
                        np.ndarray[double, ndim=1, mode="c"] mu, \
                        np.ndarray[double, ndim=1, mode="c"] inv_covariance, \
                        long ndim):
    """Wrapper to evaluate delta_theta_ij squared which is part of the gradient 
    of the objective function.

    Args:
        ndarray x: 1D array containing the current sample (ndim).
        ndarray mu: 1D array containing the centre of the guassian with
            shape (ndim).
        ndarray inv_covariance: 1D array containing the inverse covariance
            of the guassian with shape (ndim).
        long ndim: Dimension of the problem.

    Returns:
        double delta_theta_ij: Value of delta_theta_ij squared.
    """

    return delta_theta_ij(x, mu, inv_covariance, ndim)


cdef double calculate_I_ij(np.ndarray[double, ndim=1, mode="c"] x, \
                        np.ndarray[double, ndim=1, mode="c"] mu, \
                        np.ndarray[double, ndim=1, mode="c"] inv_covariance, \
                        double alpha, double weight, double ln_Pi, long ndim, \
                        double mean_shift):
    """Evaluate I_ij which is part the gradient of the objective function.

    Args:
        ndarray x: 1D array containing the current sample (ndim).
        ndarray mu: 1D array containing the centre of the guassian with
            shape (ndim).
        ndarray inv_covariance: 1D array containing the inverse covariance
            of the guassian with shape (ndim).
        double alpha: Current values of alpha (for this guassian).
        double weights: Current values of the weight (for this guassian).
        ndarray ln_Pi: Current ln posterior.
        long ndim: Dimension of the problem.
        double mean_shift: The mean of the Y values to remove that size from 
            scaling the gradient.

    Returns:
        double I_ij: Value of I_ij.
    """

    cdef double norm = alpha**(-ndim)#calculate_gaussian_normalisation(alpha, inv_covariance, ndim)
    cdef double delta_theta = delta_theta_ij(x, mu, inv_covariance, ndim)
    # if weight*norm*exp(-delta_theta/(2.0*alpha*alpha) - ln_Pi + mean_shift) > 1E8:
    # print("ln_Pi ", ln_Pi)
    # print("x ", x)
    # print("mu ", mu)
    # print("inv_covariance ", inv_covariance)
    # print("delta_theta ", delta_theta)
    # print("norm", norm)
    # print("I_ij ", weight*norm*exp(-delta_theta/(2.0*alpha*alpha) - ln_Pi + mean_shift))

    return weight*norm*exp(-delta_theta/(2.0*alpha*alpha) - ln_Pi + mean_shift)

cdef double calculate_I_i(np.ndarray[double, ndim=1, mode="c"] x, \
                np.ndarray[double, ndim=2, mode="c"] centres, \
                np.ndarray[double, ndim=2, mode="c"] inv_covariances, \
                np.ndarray[double, ndim=1, mode="c"] alphas, \
                np.ndarray[double, ndim=1, mode="c"] weights, \
                double ln_Pi, long nguassians, long ndim, double mean_shift):
    """Evaluate I_i which is part the gradient of the objective function.

    Args:
        ndarray x: 1D array containing the current sample (ndim).
        ndarray centres: 2D array containing the centres of the guassians with
            shape (nguassians, ndim).
        ndarray inv_covariances: 2D array containing the inverse covariance
            of the guassians with shape (nguassians, ndim).
        ndarray alphas: 1D array containing the current values of alpha
            with shape (nguassians).
        ndarray weights: 1D array containing the current values of the (linear) 
            weights with shape (nguassians)
        ndarray ln_Pi: Current ln posterior.
        long nguassians: Number of guassians.
        long ndim: Dimension of the problem.
        double mean_shift: The mean of the Y values to remove that size fromscaling the gradient

    Returns:
        double I_i: the value of I_i
    """

    cdef double I_i = 0.0
    cdef long i_guas

    for i_guas in range(nguassians):
        I_i += calculate_I_ij(x, centres[i_guas,:], inv_covariances[i_guas,:], \
                    alphas[i_guas], weights[i_guas], ln_Pi, ndim, mean_shift)
    # print("I_i ", I_i)
    return I_i

cdef void gradient_i1i2(np.ndarray[double, ndim=1, mode="c"] grad_alpha, \
                        np.ndarray[double, ndim=1, mode="c"] grad_beta, \
                        np.ndarray[double, ndim=2, mode="c"] X, \
                        np.ndarray[double, ndim=2, mode="c"] centres, \
                        np.ndarray[double, ndim=2, mode="c"] inv_covariances, \
                        np.ndarray[double, ndim=1, mode="c"] alphas, \
                        np.ndarray[double, ndim=1, mode="c"] weights, \
                        np.ndarray[double, ndim=1, mode="c"] Y, \
                        long nguassians, long ndim, long i1_sample, \
                        long i2_sample, \
                        np.ndarray[long, ndim=1, mode="c"] index_perm, \
                        double gamma, double mean_shift):
    """Evaluate the gradient of the objective function.

    Args:
        ndarray grad_alpha: 1D array where the gradient of alpha will be placed
            shape (nguassians).
        ndarray grad_beta: 1D array where the gradient of beta will be placed
            shape (nguassians)
        ndarray X: 2D array containing the X values shape (nsamples, ndim).
        ndarray centres: 2D array containing the centres of the guassians with
            shape (nguassians, ndim).
        ndarray inv_covariances: 2D array containing the inverse covariance
            of the guassians with shape (nguassians, ndim).
        ndarray alphas: 1D array containing the current values of alpha
            with shape (nguassians).
        ndarray weights: 1D array containing the current values of the (linear) 
            weights with shape (nguassians)
        ndarray Y: 1D array containing the X values shape (nsamples).
        long nguassians: Number of guassians.
        long ndim: Dimension of the problem.
        long i1_sample: First sample to be considered (usefull for 
            mini-batch gradient decent).
        long i2_sample: Second sample to be considered.
        ndarray index_perm: 1D array containing a random permutation of the 
            sample indexes with shape (X.shape[0]).
        double gamma: Regularisation parameter.
        double mean_shift: Mean of the Y values to remove that size 
            from scaling the gradient.

    Returns:
        None
    """

    cdef np.ndarray[double, ndim=1, mode='c'] x_i, mu_g, inv_cov_g
    cdef double I_i, I_ij, dummy
    cdef long i_sample, i_guas, i_dim, index

    x_i       = np.zeros(ndim)
    mu_g      = np.zeros(ndim)
    inv_cov_g = np.zeros(ndim)

    for i_guas in range(nguassians):
        grad_alpha[i_guas] = <double>0
        grad_beta[i_guas]  = <double>0

    for i_sample in range(i1_sample,i2_sample):
        index = index_perm[i_sample]
        for i_dim in range(ndim):
            x_i[i_dim] = X[index,i_dim]
        I_i = calculate_I_i(x_i, centres, inv_covariances, alphas, \
                  weights, Y[index], nguassians, ndim, mean_shift)

        for i_guas in range(nguassians):
            for i_dim in range(ndim):
                mu_g[i_dim] = centres[i_guas,i_dim]
                inv_cov_g[i_dim] = inv_covariances[i_guas,i_dim]

            I_ij = calculate_I_ij(x_i, mu_g, inv_cov_g, alphas[i_guas], \
                weights[i_guas], Y[index], ndim, mean_shift)

            dummy  = -<double>ndim/alphas[i_guas]
            dummy += delta_theta_ij(x_i, mu_g, inv_cov_g, ndim)/(alphas[i_guas]*alphas[i_guas]*alphas[i_guas])
            dummy *= 2*I_i*I_ij
            grad_alpha[i_guas] += dummy + alphas[i_guas]*gamma
            grad_beta[i_guas]  += 2*I_i*(I_ij - I_i*weights[i_guas])

    for i_guas in range(nguassians):
        grad_alpha[i_guas] /= <double>i2_sample - <double>i1_sample
        grad_beta[i_guas]  /= <double>i2_sample - <double>i1_sample 

    return
    
cdef double objective_function(np.ndarray[double, ndim=2, mode="c"] X, \
                        np.ndarray[double, ndim=2, mode="c"] centres, \
                        np.ndarray[double, ndim=2, mode="c"] inv_covariances, \
                        np.ndarray[double, ndim=1, mode="c"] alphas, \
                        np.ndarray[double, ndim=1, mode="c"] weights, \
                        np.ndarray[double, ndim=1, mode="c"] Y, \
                        long nguassians, long ndim, long nsamples, \
                        double gamma, double mean_shift):
    """Evaluate the scaled objective function.

    Args:
        ndarray X: 2D array containing the X values shape (nsamples, ndim).
        ndarray centres: 2D array containing the centres of the guassians with
            shape (nguassians, ndim).
        ndarray inv_covariances: 2D array containing the inverse covariance
            of the guassians with shape (nguassians, ndim).
        ndarray alphas: 1D array containing the current values of alpha
            with shape (nguassians).
        ndarray weights: 1D array containing the current values of the (linear) 
            weights with shape (nguassians).
        ndarray Y: 1D array containing the X values shape (nsamples).
        long nguassians: Number of guassians.
        long ndim: Dimension of the problem.
        double gamma: Regularisation parameter.
        double mean_shift: The mean of the Y values to remove that size from 
            scaling the gradient.

    Returns:
        double: Scaled objective function.
    """

    cdef np.ndarray[double, ndim=1, mode='c'] x_i
    cdef double I_i, reg=0.0
    cdef long i_sample, i_dim, index, i_guas

    x_i       = np.zeros(ndim)

    for i_sample in range(nsamples):
        index = i_sample
        for i_dim in range(ndim):
            x_i[i_dim] = X[index,i_dim]
        I_i = calculate_I_i(x_i, centres, inv_covariances, alphas, \
                  weights, Y[index], nguassians, ndim, mean_shift)

    for i_guas in range(nguassians):
        reg += alphas[i_guas]*alphas[i_guas]

    return I_i*I_i + 0.5*gamma*reg

class ModifiedGaussianMixtureModel(Model):    
    """ModifiedGaussianMixtureModel (MGMM) to approximate the log_e posterior by
    a modified Gaussian mixture model.
    """

    def __init__(self, long ndim, list domains not None, hyper_parameters=[3,1E-8,None,None,None]):        
        """Constructor setting the hyper parameters and domains of the model
        of the MGMM which models the posterior as a group of Gaussians.

        Args:
            long ndim: Dimension of the problem to solve.
            list domains: A list of length 1 with the range of scale parameter
                of the covariance matrix, i.e. the range of alpha, where
                C' = alpha * C_samples, and C_samples is the diagonal of the covariance in the samples in each cluster.
            hyper_parameters: A list of length 2, the first of which should be 
                nummber of clusters and the second is the regularisation 
                parameter gamma.

        Raises:
            ValueError: If the hyper_parameters list is not length 5.
            ValueError: If the length of domains list is not 1.
            ValueError: If the ndim is not positive.
        """

        if len(hyper_parameters) != 5:
            raise ValueError("ModifiedGaussianMixtureModel model hyper_parameters list " +
                "shoule be length 5.")
        if len(domains) != 1:
            raise ValueError("ModifiedGaussianMixtureModel model domains list should " +
                "be length 1.")
        if ndim < 1:
            raise ValueError("ndim must be greater than 0.")

        if hyper_parameters[0] < 1:
            raise ValueError("nguassians must be a positive integer")

        self.ndim                = ndim
        self.alpha_domain        = domains[0]
        self.nguassians          = hyper_parameters[0]
        self.gamma               = hyper_parameters[1]
        self.beta_weights        = np.zeros(self.nguassians)
        self.alphas              = np.ones(self.nguassians)
        self.centres             = np.zeros((self.nguassians,self.ndim))
        self.inv_covariance      = np.ones((self.nguassians,self.ndim))
        self.centres_inv_cov_set = False
        if hyper_parameters[2] == None:
            self.learning_rate   = 0.1
        else:
            self.learning_rate   = hyper_parameters[2]
        if hyper_parameters[3] == None:
            self.max_iter        = 50
        else:
            self.max_iter        = hyper_parameters[3]
        if hyper_parameters[4] == None:
            self.nbatch          = 100
        else:
            self.nbatch          = hyper_parameters[4]
        self.verbose             = False
        self.fitted              = False

    def is_fitted(self):
        """Specify whether model has been fitted.
        
        Args: 
            None.
            
        Return:
            Boolean specifying whether the model has been fitted.
        """

        return self.fitted

    def set_weights(self, np.ndarray[double, ndim=1, mode="c"] weights_in):
        """Set the weights of the Gaussians.
        
        The weights are the softmax of the betas (without normalisation), i.e.
        the betas are the log_e of the weights.

        Args:
            ndarray weights_in: 1D array containing the weights (no need to     
                normalise) with shape (nguassians).

        Raises:
            ValueError: If the input array length is not nguassians.
            ValueError: If the input array contains a NaN.
            ValueError: If at least one of the weights is negative.
            ValueError: If the sum of the weights is too close to zero.
        """
        if weights_in.size != self.nguassians:
            raise ValueError("Weights must have length nguassians")

        for i_guas in range(self.nguassians):
            if np.isnan(weights_in[i_guas]):
                raise ValueError("Weights contains a NaN")
            if weights_in[i_guas] < 0.0:
                raise ValueError("Weights must be non-negative")

        if np.sum(weights_in) < 1E-8:
            raise ValueError("At least one weight must be non-negative")
            
        with np.errstate(divide="ignore"):
            self.beta_weights = np.log(weights_in)
            
        return

    def set_alphas(self, np.ndarray[double, ndim=1, mode="c"] alphas_in):
        """Set the alphas (i.e. scales).

        Args:
            ndarray alphas_in: 1D array containing the alpha scalings with 
                shape (nguassians)

        Raises:
            ValueError: If the input array length is not nguassians.
            ValueError: If the input array contains a NaN.
            ValueError: If at least one of the alphas not positive.
        """
        if alphas_in.size != self.nguassians:
            raise ValueError("alphas must have length nguassians")

        for i_guas in range(self.nguassians):
            if np.isnan(alphas_in[i_guas]):
                raise ValueError("alphas contains a NaN")
            if alphas_in[i_guas] <= 0.0:
                raise ValueError("alphas must be positive")

        self.alphas = alphas_in.copy()
        
        return

    def set_centres(self, np.ndarray[double, ndim=2, mode="c"] centres_in):
        """Set the centres of the Gaussians.

        Args:
            ndarray centres_in: 2D array containing the centres with
                shape (ndim, nguassians).

        Raises:
            ValueError: If the input array is not the correct shape.
            ValueError: If the input array contains a NaN.
        """
        
        if centres_in.shape[0] != self.nguassians \
            or centres_in.shape[1] != self.ndim:
            raise ValueError("centres must be shape (nguassians,ndim).")

        for i_guas in range(self.nguassians):
            for i_dim in range(self.ndim):
                if np.isnan(centres_in[i_guas,i_dim]):
                    raise ValueError("Centres contains a NaN")

        self.centres = centres_in.copy()

        return

    def set_inv_covariance(self, np.ndarray[double, ndim=2, mode="c"] 
                                 inv_covariance_in):
        """Set the inverse covariance of the Gaussians.

        Args:
            ndarray inv_covariance_in: 2D array containing the centres
                with shape (ndim, nguassians)

        Raises:
            ValueError: If the input array is not the correct shape.
            ValueError: If the input array contains a NaN.
            ValueError: If the input array contains a number that is 
                not positive.
        """
        
        if inv_covariance_in.shape[0] != self.nguassians or inv_covariance_in.shape[1] != self.ndim:
            raise ValueError("inv_covariance must be shape (nguassians,ndim)")

        for i_guas in range(self.nguassians):
            for i_dim in range(self.ndim):
                if np.isnan(inv_covariance_in[i_guas,i_dim]):
                    raise ValueError("inv_covariance contains a NaN")
                if inv_covariance_in[i_guas,i_dim] <= 0.0:
                    raise ValueError("inv_covariance contains a number that " +
                                     "is not positive")

        self.inv_covariance = inv_covariance_in.copy()

        return

    def set_centres_and_inv_covariance(self, \
        np.ndarray[double, ndim=2, mode="c"] centres_in,\
        np.ndarray[double, ndim=2, mode="c"] inv_covariance_in):
        """Set the centres and inverse covariance of the Gaussians.

        Args:
            ndarray centres_in: 2D array containing the centres
                with shape (ndim, nguassians).
            ndarray inv_covariance_in: 2D array containing the centres
                with shape (ndim, nguassians).

        Raises:
            ValueError: If the input arrays are not the correct shape.
            ValueError: If the input arrays contain a NaN.
            ValueError: If the input covariance contains a number that is 
                not positive.
        """

        if centres_in.shape[0] != self.nguassians or centres_in.shape[1] != self.ndim:
            raise ValueError("centres must be shape (nguassians,ndim)")

        for i_guas in range(self.nguassians):
            for i_dim in range(self.ndim):
                if np.isnan(centres_in[i_guas,i_dim]):
                    raise ValueError("Centres contains a NaN")

        if inv_covariance_in.shape[0] != self.nguassians or inv_covariance_in.shape[1] != self.ndim:
            raise ValueError("inv_covariance must be shape (nguassians,ndim)")

        for i_guas in range(self.nguassians):
            for i_dim in range(self.ndim):
                if np.isnan(inv_covariance_in[i_guas,i_dim]):
                    raise ValueError("inv_covariance contains a NaN")
                if inv_covariance_in[i_guas,i_dim] <= 0.0:
                    raise ValueError("inv_covariance contains a number that is not positive")

        self.centres = centres_in.copy()
        self.inv_covariance = inv_covariance_in.copy()
        self.centres_inv_cov_set = True
        return

    def fit(self, np.ndarray[double, ndim=2, mode="c"] X, 
            np.ndarray[double, ndim=1, mode="c"] Y):
        """Fit the parameters of the model as follows.
                
            If centres and inv_covariances not set:
                1) Find clusters using the k-means clustering from 
                   scikit learn.
                2) Use the samples in the clusters to find the centres and 
                  covariance matricies.
            
            Then minimize the objective function using the gradients and 
            mini-batch stochastic descent.

            Args:
                X: 2D array of samples of shape (nsamples, ndim).
                Y: 1D array of target log_e posterior values for each sample in 
                    X of shape (nsamples).
            
            Returns:
                Boolean specifying whether fit successful.

            Raises:
                ValueError if the first dimension of X is not the same as Y.
                ValueError if the first dimension of X is not the same as Y.
                ValueError if the second dimension of X is not the same as ndim.
        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y sizes are not the same")

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim")

        cdef np.ndarray[double, ndim=2, mode='c'] centres         = self.centres
        cdef np.ndarray[double, ndim=2, mode='c'] inv_covariances = self.inv_covariance
        cdef np.ndarray[double, ndim=1, mode='c'] alphas          = self.alphas
        cdef np.ndarray[double, ndim=1, mode='c'] betas           = self.beta_weights
        cdef np.ndarray[double, ndim=1, mode='c'] cluster_count
        cdef np.ndarray[double, ndim=1, mode='c'] weights
        cdef np.ndarray[double, ndim=1, mode='c'] grad_alpha
        cdef np.ndarray[double, ndim=1, mode='c'] grad_beta
        cdef np.ndarray[long, ndim=1, mode='c'] index_perm

        cdef double gamma = self.gamma, learning_rate = self.learning_rate
        cdef double alpha_lower_bound = self.alpha_domain[0], alpha_upper_bound = self.alpha_domain[1]
        cdef double mean_shift = np.mean(Y)
        cdef long i_dim, i_guas, i_sample, i_iter, i_batch, i1_sample, i2_sample
        cdef long ndim = self.ndim, nguassians = self.nguassians, nsamples = X.shape[0]
        cdef long max_iter = self.max_iter, nbatch = min(self.nbatch,nsamples)
        cdef bint keep_going = True

        if not self.centres_inv_cov_set:
            # scale data
            scaler = preprocessing.StandardScaler(copy=True).fit(X)
            X_scaled = scaler.transform(X)

            # set up with k-means clustering
            kmeans = KMeans(n_clusters=nguassians, random_state=0).fit(X_scaled)

            cluster_count = np.zeros(nguassians)

            for i_sample in range(nsamples):
                i_guas = kmeans.labels_[i_sample]
                cluster_count[i_guas] += 1

                for i_dim in range(ndim):
                    centres[i_guas, i_dim]         += X[i_sample,i_dim]
                    inv_covariances[i_guas, i_dim] += X[i_sample,i_dim]*X[i_sample,i_dim]

            for i_guas in range(nguassians):
                for i_dim in range(ndim):
                    centres[i_guas,i_dim]         = centres[i_guas,i_dim]/cluster_count[i_guas]
                    inv_covariances[i_guas,i_dim] = inv_covariances[i_guas,i_dim]/cluster_count[i_guas] \
                                                    - centres[i_guas,i_dim]*centres[i_guas,i_dim]
                    inv_covariances[i_guas,i_dim] = 1.0/inv_covariances[i_guas,i_dim]
        if self.verbose:
            print("centres : ", centres)
            print("inv_covariances : ", inv_covariances)
        # randomally incialise the parameters
        alphas[:] = np.random.lognormal(sigma=0.25,size=nguassians)
        betas[:]  = np.random.randn(nguassians)
        grad_alpha = np.zeros(nguassians)
        grad_beta = np.zeros(nguassians)
        if self.verbose:
            print("iteration : ", 0, "param ", alphas, beta_to_weights(betas, nguassians))
            print("objective function :", objective_function(X, centres, \
                                        inv_covariances, alphas, beta_to_weights(betas, nguassians), \
                                        Y, nguassians, ndim, nsamples, gamma, mean_shift))

        for i_guas in range(nguassians):
            if alphas[i_guas] < alpha_lower_bound:
                alphas[i_guas] = alpha_lower_bound
            if alphas[i_guas] > alpha_upper_bound:
                alphas[i_guas] = alpha_upper_bound

        index_perm = np.random.permutation(nsamples)

        i_iter = 0
        while(keep_going):
            for i_batch in range(nbatch):
                i1_sample = i_batch*nsamples//nbatch
                if i_batch == nbatch-1:
                    i2_sample = nsamples
                else:
                    i2_sample = (i_batch+1)*nsamples//nbatch
                # calculate gradients
                weights = beta_to_weights(betas, nguassians)
                gradient_i1i2(grad_alpha, grad_beta, X, centres, \
                              inv_covariances, alphas, weights, Y, \
                              nguassians, ndim, i1_sample, \
                              i2_sample, index_perm, gamma, mean_shift)
                # print("grads ", grad_alpha, grad_beta)
                # update parameters
                for i_guas in range(nguassians):
                    alphas[i_guas] -= learning_rate*grad_alpha[i_guas]
                    betas[i_guas]  -= learning_rate*grad_beta[i_guas]
                # print(" ", alphas, betas)

                for i_guas in range(nguassians):
                    if alphas[i_guas] < alpha_lower_bound:
                        alphas[i_guas] = alpha_lower_bound
                    if alphas[i_guas] > alpha_upper_bound:
                        alphas[i_guas] = alpha_upper_bound

            # check stopping criteria
            i_iter += 1
            if self.verbose:
                print("iteration : ", i_iter, "param ", alphas, beta_to_weights(betas, nguassians))
                print("objective function :", objective_function(X, centres, \
                                            inv_covariances, alphas, weights, \
                                            Y, nguassians, ndim, nsamples, gamma, mean_shift))
            if i_iter >= max_iter:
                keep_going = False

        self.fitted = True
        return

    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the hight of the posterior at point x.
        
        Args: 
            x: 1D array of sample of shape (ndim) to predict posterior value.

        Return:
            Predicted posterior value.
        """

        cdef np.ndarray[double, ndim=2, mode="c"] mus = self.centres
        cdef np.ndarray[double, ndim=2, mode="c"] inv_covariances = self.inv_covariance
        cdef np.ndarray[double, ndim=1, mode="c"] alphas = self.alphas
        cdef np.ndarray[double, ndim=1, mode="c"] weights 

        cdef long i_guas, nguassians = self.nguassians, ndim = self.ndim
        cdef double value = 0.0

        weights = beta_to_weights(self.beta_weights, nguassians)

        for i_guas in range(nguassians):
            value += evaluate_one_guassian(x, mus[i_guas,:], inv_covariances[i_guas,:], \
                                           alphas[i_guas], weights[i_guas], ndim)
        return log(value)
