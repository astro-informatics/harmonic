import abc
import numpy as np
cimport numpy as np
from libc.math cimport log, exp, sqrt
import scipy.special as sp
import scipy.optimize as so

class Model(metaclass=abc.ABCMeta):
    """Base abstract class for posterior model.
    
    All inherited models must implement the abstract constructor, fit and 
    predict methods.
    """

    @abc.abstractmethod
    def __init__(self, int ndim, list domains not None, hyper_parameters=None):
        """Constructor setting the hyper parameters and domains of the model.
        
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


cdef double HyperSphereObjectiveFunction(double R_squared, X, Y, \
                              centre, inv_covariance):
    """ Evaluates the ojective function with the HyperSphere model
        this is the varience of the estimator subject to a linear transformation
        that does not depend on the radius of the sphere (the bit being fitted).

    Args:
        double R_squared: the radius of the hyper sphere squared
        X: 2D numpy.ndarray containing the samples 
            with shape (nsamples, ndim) and dtype double.
        Y: 1D numpy.ndarray containing the log_e posterior 
            values with shape (nsamples) and dtype double.
        centre: 1D numpy.ndarray containing the centre of the sphere 
            with shape (ndim) and dtype double.
        ln_posterior: 1D numpy.ndarray containing the diagonal of the
            inverse covariance matrix
            values with shape (ndim) and dtype double.

    Return:
        The value of the objective function
    """

    cdef np.ndarray[double, ndim=2, mode="c"] X_here = X
    cdef np.ndarray[double, ndim=1, mode="c"] Y_here = Y, \
        centre_here = centre, \
        inv_covariance_here = inv_covariance

    cdef int i_dim, i_sample, ndim = X.shape[1], nsample = X.shape[0]
    cdef double objective = 0.0, distance, mean_shift = np.mean(Y)
    cdef double ln_volume = ndim*log(R_squared)/2 # Parts that do not depend on R is ignored

    for i_sample in range(nsample):
        distance_squared = 0.0
        for i_dim in range(ndim):
            distance_squared += \
                (X_here[i_sample,i_dim] - centre_here[i_dim]) \
                * (X_here[i_sample,i_dim] - centre_here[i_dim]) \
                * inv_covariance_here[i_dim]
        if distance_squared < R_squared:
            objective += exp( 2*(mean_shift - Y[i_sample]) ) 

    objective = exp(-2*ln_volume)*objective/nsample
    # objective = exp(-2*ln_one_over_volume-2*mean_shift)*objective/nsample

    return objective


class HyperSphere(Model):

    def __init__(self, long ndim_in, list domains not None, hyper_parameters=None):
        """ constructor setting the hyper parameters of the model

        Args:
            long ndim_in: The dimension of the problem to solve
            list domains: A list of length 1 containing a 1D array
                of length 2 containing the lower and upper bound of the
                radius of the hyper sphere
            hyper_parameters: Should not be set as there are no hyper parameters
                for this model

        Raises:
            ValueError: If the hyper_parameters variable is not None
            ValueError: If the length of domains list is not one
            ValueError: If the ndim_in is not positive
        """
        if hyper_parameters != None:
            raise ValueError("HyperSphere model has no hyper parameters.")
        if len(domains) != 1:
            raise ValueError("HyperSphere model domains list should " +
                "be length 1.")
        if ndim_in < 1:
            raise ValueError("The dimension must be greater then 1")

        self.ndim               = ndim_in
        self.centre_set         = False
        self.centre             = np.zeros((ndim_in))
        self.inv_covariance_set = False
        self.inv_covariance     = np.ones((ndim_in))
        self.R_domain           = domains[0]
        self.set_R(np.mean(self.R_domain))
        self.fitted             = False

    def set_R(self, double R):
        """ sets the radius of the hyper sphere and calculates the volume of the sphere

        Args:
            double R: The radius sphere

        Raises:
            ValueError: If the radius is a NaN
            ValueError: If the Raises is not positive
        """

        if ~np.isfinite(R):
            raise ValueError("Radius is a NaN.")
        if R <= 0.0:
            raise ValueError("Radius must be positive.")

        self.R = R
        self.set_precompucted_values()
        
        return

    def set_precompucted_values(self):
        """ precomputes volume of the hyper sphere (scaled ellipse)
            and the squared radius

        Args:
            None

        Raises:
            None

        Returns:
            None
        """
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
        """Sets the centre of the hyper sphere

        Args:
            centre_in: 1D numpy.ndarray containing the centre of sphere 
                with shape (ndim) and dtype double.

        Raises:
            ValueError: If the length of the centre array is not the same as
                ndim
            ValueError: If the centre array contains a NaN
        """

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

        """Sets the centre of the hyper sphere

        Args:
            inv_covariance_in: 1D numpy.ndarray containing the diagonal of 
                inverse convarience matrix that defines the ellipse
                with shape (ndim) and dtype double.

        Raises:
            ValueError: If the length of the inv_covariance array is not the same as
                ndim
            ValueError: If the inv_covariance array contains a NaN
            ValueError: If the inv_covariance array contains a value that is not positive
        """

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

        self.set_R(sqrt(result.x))

        self.fitted = result.success

        return result.success

    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the hight of the log_e posterior at point x

        Args: 
            x: 1D array of sample of shape (ndim) to predict posterior value.
        
        Return:
            Predicted posterior value.
        """

        x_minus_centre = x - self.centre        
        
        distance_squared = \
            np.dot(x_minus_centre, x_minus_centre*self.inv_covariance)

        if distance_squared < self.R_squared:
            return self.ln_one_over_volume
        else:
            return -np.inf
    
cdef KernelDensityEstimate_set_grid(dict grid, \
                                    np.ndarray[double, ndim=2, mode="c"] X, 
                                    np.ndarray[double, ndim=2, mode="c"] start_end, \
                                    np.ndarray[double, ndim=1, mode="c"] inv_scales, \
                                    long ngrid, double D):
    """ Creates a dictionary that allows a fast way to find the indexes of samples
        in a pixel in a grid where the pixel sizes are the diameter of the hyper
        spheres placed at each sample

    Args:
        dict grid: Empty dictionary where the list of the sample index will be placed.
            The key is a index of the grid (c type ordering) and the value is a list
            containing the indexes in the sample array of all the samples in that index
        X: 2D array of samples of shape (nsamples, ndim).
        Y: 1D array of target log_e posterior values for each sample in X 
            of shape (nsamples).
        start_end:  2D array of the lowest and highest sample in each dimension (ndim,2)
        inv_scales: 1D array of the 1.0/delta_x_i where delta_x_i is the difference between the
            max and min of the sample in dimension i
        long ngrid: Number of pixels in each dimension in the grid
        double D: The diameter of the hyper sphere

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
            # print(i_dim, X_in[i_sample,i_dim], start_end[i_dim,0], inv_scales[i_dim], (X_in[i_sample,i_dim]-start_end[i_dim,0])*inv_scales[i_dim], sub_index, ngrid, index)
        if index in grid:
            grid[index].append(i_sample)
        else:
            grid[index] = [i_sample]

cdef KernelDensityEstimate_loop_round_and_search(long index, long i_dim, long ngrid,\
                                                 long ndim, dict grid, \
                                                 np.ndarray[double, ndim=2, mode="c"] samples, \
                                                 np.ndarray[double, ndim=1, mode="c"] x, \
                                                 np.ndarray[double, ndim=1, mode="c"] inv_scales, \
                                                 double distance, long *count):
    """ This is a recursive function that calls itself inorder to call the search_in_pixel 
        function on one pixel behind and infront of the pixel x is in for each dimension

    Args:
        long index: The current pixel we are looking at
        long i_dim: The dimension we are doing the current moving forward and backward in
        long ngrid: Number of pixels in each dimension in the grid
        long ndim: The dimension of the problem
        dict grid: The dictionary with information on which samples are in which pixel. The key 
            is an index of the grid (c type ordering) and the value is a list
            containing the indexes in the sample array of all the samples in that index
        samples: 2D array of samples of shape (nsamples, ndim).
        x: 1D array of the position we are evaluating the pridiction for
        inv_scales: 1D array of the 1.0/delta_x_i where delta_x_i is the difference between the
            max and min of the sample in dimension i
        double distance: The diameter of the hyper sphere
        long * count: a pointer to the count integer that counts how many hyper spheres the postion
            x falls inside

    Returns:
        None
    """
    # this does create looping boundry conditions but doesn't matter in searching
    # it will simply slow things down very very slightly 
    # (probably less then dealing with it will!)
    if i_dim >= 1:
        for iter_i_dim in range(-1,2):
            KernelDensityEstimate_loop_round_and_search(index+iter_i_dim*ngrid**(i_dim-1), i_dim - 1, ngrid, ndim, grid, samples, x, inv_scales, distance, count)
    else:
        KernelDensityEstimate_search_in_pixel(index, grid, samples, x, inv_scales, distance, count)

cdef KernelDensityEstimate_search_in_pixel(long index, dict grid, \
                                           np.ndarray[double, ndim=2, mode="c"] samples, \
                                           np.ndarray[double, ndim=1, mode="c"] x, \
                                           np.ndarray[double, ndim=1, mode="c"] inv_scales, \
                                           double distance, long *count):
    """ Examines all samples that are in the current pixel and counts how
        many of those position x falls inside

    Args:
        long index: The current pixel we are looking at
        long ndim: The dimension of the problem
        dict grid: The dictionary with information on which samples are in which pixel. The key 
            is an index of the grid (c type ordering) and the value is a list
            containing the indexes in the sample array of all the samples in that index
        samples: 2D array of samples of shape (nsamples, ndim).
        x: 1D array of the position we are evaluating the prediction for
        inv_scales: 1D array of the 1.0/delta_x_i where delta_x_i is the difference between the
            max and min of the sample in dimension i
        double distance: The diameter of the hyper sphere
        long * count: a pointer to the count integer that counts how many hyper spheres the postion
            x falls inside

    Returns:
        None
    """
    if index in grid:
        for sample_index in grid[index]:
            # do what I did before
            length = np.dot((x-samples[sample_index,:])*inv_scales, (x-samples[sample_index,:])*inv_scales)
            if length<distance:
                count[0] += 1
    return

class KernelDensityEstimate(Model):

    def __init__(self, int ndim, list domains not None, hyper_parameters=None):
        """ constructor setting the hyper parameters and domains of the model

        Args:
            long ndim: The dimension of the problem to solve
            list domains: A list of length 0
            hyper_parameters: A list of length 1 which should be the diameter in scaled
                units of the hyper spheres to use in the Kernel Density Estimate


        Raises:
            ValueError: If the hyper_parameters list is not length 1
            ValueError: If the length of domains list is not 0
            ValueError: If the ndim_in is not positive
        """
        if len(hyper_parameters) != 1:
            raise ValueError("Kernel Density Estimate hyper parameter list should be length 1.")
        if len(domains) != 0:
            raise ValueError("Kernel Density Estimate domains list should be length 0.")
        if ndim < 1:
            raise ValueError("The dimension must be greater then 1")

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
        """ sets the scales of the hyper spheres based on the min
            and max sample in each dimension

        Args:
            X: 2D array of samples of shape (nsamples, ndim).

        Returns:
            None

        Raises:
            ValueError if the second dimension of X is not the same as ndim
        """

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim")

        cdef int i_dim

        for i_dim in range(self.ndim):
            self.inv_scales[i_dim]  = 1.0/(X[:,i_dim].max() - X[:,i_dim].min())
            self.start_end[i_dim,0] = X[:,i_dim].min()
            self.start_end[i_dim,1] = X[:,i_dim].max()

        self.inv_scales_squared = self.inv_scales**2

        self.scales_set = True

        return

    def precompute_normalising_factor(self, np.ndarray[double, ndim=2, mode="c"] X):
        """precomputes the log_e normalisation factor of the density estimation

        Args:
            X: 2D array of samples of shape (nsamples, ndim).

        Raises:
            ValueError if the second dimension of X is not the same as ndim
        """

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim")

        cdef double ln_volume, det_covarience = 1.0

        for i_dim in range(self.ndim):
            det_covarience *= 1.0/self.inv_scales[i_dim]

        ln_volume = ((self.ndim/2)*log(np.pi) + self.ndim*log(self.D/2) - sp.gammaln(self.ndim/2+1) + log(det_covarience))
         

        self.ln_norm = log(<double>X.shape[0]) + ln_volume
        pass



    def fit(self, np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=1, mode="c"] Y):
        """Fit the parameters of the model by:
            1) Setting the scales of the model from the samples
            2) creating the dictionary containing all the information on which samples are in 
               which pixel in a grid where each pixel size is the same as the diameter of the
               hyper spheres to be placed on each sample. The key is an index of the grid 
               (c type ordering) and the value is a list containing the indexes in the sample 
               array of all the samples in that index
            3) Precomputes the normalisation factor

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

        self.samples = X.copy() # TODO consider functionality for shallow copy to save mem

        self.set_scales(self.samples)

        #set dictionary 
        KernelDensityEstimate_set_grid(self.grid, self.samples, self.start_end, self.inv_scales, self.ngrid, self.D)

        self.precompute_normalising_factor(self.samples)

        self.fitted = True

        return True

    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the hight of the posterior at point x.

        Args: 
            x: 1D array of sample of shape (ndim) to predict posterior value.
        
        Return:
            Predicted posterior value.
        """
        cdef np.ndarray[double, ndim=2, mode="c"] samples    = self.samples,   start_end = self.start_end
        cdef np.ndarray[double, ndim=1, mode="c"] inv_scales = self.inv_scales
        cdef long i_sample, i_dim, sub_index, index, nsamples = samples.shape[0], ndim = self.ndim, ngrid = self.ngrid
        cdef double inv_diam = 1.0/self.D, distance = self.distance
        cdef dict grid = self.grid

        cdef long count = 0
        #find out the pixel that the sample is in
        index = 0
        for i_dim in range(ndim):
            sub_index = <long>((x[i_dim]-start_end[i_dim,0])*inv_scales[i_dim]*inv_diam) + 1
            index += sub_index*ngrid**i_dim

        KernelDensityEstimate_loop_round_and_search(index, i_dim, ngrid, ndim, grid, samples, x, inv_scales, distance, &count)

        return log(count) - self.ln_norm


class ModifiedGaussianMixtureModel(Model):

    def __init__(self, int ndim, list domains not None, hyper_parameters=None):
        """ constructor setting the hyper parameters and domains of the model
        """

    def fit(self, np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=1, mode="c"] Y):
        """Fit the parameters of the model
        """
        return

    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the hight of the posterior at point x
        """
        return
