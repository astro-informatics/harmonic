import numpy as np
from libc.math cimport sin
import abc


class PluginBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load(self, input):
        """Retrieve data from the input source
        and return an object.
        """

    @abc.abstractmethod
    def save(self, output, data):
        """Save the data object to the output."""


cdef float my_sin(float x):
    return sin(x)



class RegisteredImplementation(PluginBase):

    def __init__(self):
        cdef int i

    def load(self):
        cdef int i
        cdef float f
        for self.i in range(100):
            f = my_sin(4.5)
        return


    def save(self, output, data):
        return output.write(data)