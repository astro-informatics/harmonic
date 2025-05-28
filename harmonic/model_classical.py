import numpy as np
from harmonic import model_abstract as mda
from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class HistogramModel(mda.Model):
    """Class to compute the histogram model for the given value and log prior.

    Args:
        ndim (float): Prior probability.
        nbins (int): Number of bins to use for the histogram.

    """
    ndim: int
    nbins: int = 20
    
    def __post_init__(self):
        if self.ndim < 1:
            raise ValueError("Dimension must be greater than 0.")
        if self.nbins < 2:
            raise ValueError("Number of bins must be greater than 1.")
        
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
        """Predict the value of the histogram model."""
         
        value = np.atleast_1d(value)
        bin_indices = [np.digitize(v, edges) - 1 for v, edges in zip(value, self.bin_edges)]
        
        if np.logical_or(np.asarray(bin_indices) < 0, np.asarray(bin_indices) > self.nbins).any():
            raise ValueError(
                f"Value {value} is outside the range of the histogram bins"
                f"This means that it is outside of the support of your samples"
            )
        
        bin_indices = tuple(bin_indices)
        prob = self.hist[bin_indices]
        return jnp.log(prob)
