import numpy as np
import tmp_poly_class_example as tmp
import timeit
import abc
import hyper_sphere


class PluginBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load(self, input):
        """Retrieve data from the input source
        and return an object.
        """

    @abc.abstractmethod
    def save(self, output, data):
        """Save the data object to the output."""

def my_sin(x):
    return np.sin(x)


class RegisteredImplementation(PluginBase):

    def load(self):
        for i in range(100):
            f = self.my_sin(4.5)
        return 

    def my_sin(self, x):
        return my_sin(x)


    def save(self, output, data):
        return output.write(data)

if __name__ == '__main__':
    print(hyper_sphere.HyperSphere([0.1,1.0]))

    print('Subclass:', issubclass(RegisteredImplementation,
                                  PluginBase))
    print('Instance:', isinstance(RegisteredImplementation(),
                                  PluginBase))
    RegisteredImplementation()
    n_loop = 1000
    print(n_loop)
    # print(tmp.first_function(3))

    test_python  = RegisteredImplementation()
    test_cython  = tmp.RegisteredImplementation()
    # test_cython2 = hm.lone_class()
    print(timeit.timeit(test_python.load, 'gc.enable()', number=n_loop)/n_loop)
    print(timeit.timeit(test_cython.load, 'gc.enable()', number=n_loop)/n_loop)
    # print(timeit.timeit(test_cython2.load, 'gc.enable()', number=n_loop)/n_loop)
    # print(timeit.timeit(hm.first_function, 'gc.enable()', number=n_loop)/n_loop)
