import abc
import cloudpickle
import numpy as np


class Model(metaclass=abc.ABCMeta):
    """Base abstract class for posterior model.

    All inherited models must implement the abstract constructor, fit and
    predict methods.

    """

    @abc.abstractmethod
    def __init__(self, ndim: int):
        """Constructor setting the hyper-parameters and domains of the model.

        Must be implemented by derived class (currently abstract).

        Args:

            ndim (int): Dimension of the problem to solve.
        """

    @abc.abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit the parameters of the model.

        Must be implemented by derived class (currently abstract).

        Args:

            X (np.ndarray[nsamples, ndim]): Sample x coordinates.

            Y (np.ndarray[nsamples]): Target log_e posterior values for each
                sample in X.

        """

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the value of the posterior at point x.

        Must be implemented by derived class (since abstract).

        Args:

            x (np.ndarray): Sample of shape (ndim) at which to
                predict posterior value.

        Returns:

            (np.ndarray): Predicted log_e posterior value.

        """

    def is_fitted(self) -> bool:
        """Specify whether model has been fitted.

        Returns:

            (bool): Whether the model has been fitted.

        """

        return self.fitted

    def serialize(self, filename):
        """Serialize Model object.

        Args:

            filename (string): Name of file to save model object.

        """

        file = open(filename, "wb")
        cloudpickle.dump(self, file)
        file.close()

        return

    @classmethod
    def deserialize(self, filename):
        """Deserialize Model object from file.

        Args:

            filename (string): Name of file from which to read model object.

        Returns:

            (Model): Model object deserialized from file.

        """
        file = open(filename, "rb")
        model = cloudpickle.load(file)
        file.close()

        return model
