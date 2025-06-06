import numpy as np
from harmonic import model_abstract as mda
import jax.numpy as jnp

class HistogramModel(mda.Model):
    """Histogram model where a density is fitted with a normalized histogram. 
        To be used with the SDDR computation only.

    Args:
        ndim (float): Prior probability.
        nbins (int): Number of bins to use for the histogram.

    """
    
    def __init__(self, ndim: int, nbins: int = 20):
        if ndim < 1:
            raise ValueError("Dimension must be greater than 0.")
        if nbins < 2:
            raise ValueError("Number of bins must be greater than 1.")
        
        self.ndim = ndim
        self.nbins = nbins
        self.fitted = False
    
    def fit(self, X: jnp.ndarray):
        """Fit the histogram model to the data.
        
        Args:
            X (jnp.ndarray (nsamples, ndim)): Training samples.
        
        
        Raises:

            ValueError: Raised if the second dimension of X is not the same as ndim.

        """
        
        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim.")
        
        self.hist, self.bin_edges = np.histogramdd(
            sample=X,
            bins=self.nbins,
            density=True,
        )
        
        self.fitted = True
        
        return 
    
    def predict(self, value: float | np.ndarray) -> float:
        """Predict the value of the histogram model.
        Args: 
            value (float | np.ndarray): Value to predict the histogram for.
        Returns:
            float: Log probability of the value in the histogram.
        Raises:
            ValueError: Raised if the value is outside the range of the histogram bins.
        """
         
        value = np.atleast_1d(value)
        bin_indices = [np.digitize(v, edges) - 1 for v, edges in zip(value, self.bin_edges)]
        
        if np.logical_or(np.asarray(bin_indices) < 0, np.asarray(bin_indices) >= self.nbins).any():
            raise ValueError(
                f"Value {value} is outside the range of the histogram bins."
                f"This means that it is outside of the support of your samples."
            )
        
        bin_indices = tuple(bin_indices)
        prob = self.hist[bin_indices]
        return jnp.log(prob)
