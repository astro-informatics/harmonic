import abc
import numpy as np
cimport numpy as np


class ModelBase(metaclass=abc.ABCMeta):
    """ Base class for model
    """

    @abc.abstractmethod
    def __init__(self, ranges, hyper_parameters=None):
        """ constructor setting the hyper parameters of the model
        """

    @abc.abstractmethod
    def fit(self, np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=1, mode="c"] Y):
        """Fit the parameters of the model
        """

    @abc.abstractmethod
    def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
        """Use model to predict the hight of the posterior at point x
        """

