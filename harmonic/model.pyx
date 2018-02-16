import abc
import numpy as np
cimport numpy as np


class Model(metaclass=abc.ABCMeta):
    """Base class for model
    
    All inherited models must implement the abstract constructor and fit and predict methods.
    """

    @abc.abstractmethod
    def __init__(self, domains, hyper_parameters=None):
        """Constructor setting the hyper parameters of the model
        
        Must be implemented by derivied class (currently abstract).
        
        Args: 
            domains: List of domains for each parameter of model.  Each domain 
                is a list of length two, specifying a lower and upper bound.  
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



class HyperSphere(Model):

    def __init__(self, ranges, hyper_parameters=None):
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
    


class KernelDensityEstimate(Model):

    def __init__(self, ranges, hyper_parameters=None):
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

    def __init__(self, ranges, hyper_parameters=None):
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
        