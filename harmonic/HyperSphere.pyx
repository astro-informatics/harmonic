import abc
import numpy as np
cimport numpy as np
import model_base_class as model_base

class LocalBaseClass:
    pass


@model_base.ModelBase.register
class HyperSphere(LocalBaseClass):

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
