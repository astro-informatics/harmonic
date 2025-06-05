import numpy as np
from tqdm import tqdm
from copy import deepcopy
from harmonic import model_abstract as mda
import warnings

class sddr:
    """Class to compute the Bayes factor between two models with the Savage-Dicky density ratio.
       
    Args:
        model (Model): An instance of a posterior model class that has not been fitted.
        samples (np.ndarray): The marginalised samples to use for the Bayes factor computation with the SDDR.
            Shape should be (nsamples, ndim) where nsamples is the number of samples
            and ndim is the number of dimensions of the model.
    """
    
    def __init__(self, model: mda.Model, samples: np.ndarray[tuple[np.float64, np.float64]]):
        self.model = model
        self.samples = samples

        if self.model.is_fitted():
            raise ValueError("Model is already fitted, pass in an unfitted model.")
        
        if len(self.samples.shape) == 1:
            self.samples = self.samples.reshape(-1, 1)
        elif self.model.ndim != self.samples.shape[1]:
            raise ValueError(
                f"Model ndim {self.model.ndim} does not match samples shape {self.samples.shape[1]}."
            )
            
        if self.model.ndim == 1 and self.model.__class__.__name__ == 'RealNVPModel':
            raise ValueError(
                "RealNVPModel is not supported for 1D SDDR computation. Use HistogramModel or RQSplineModel instead."
            )
        
        if not hasattr(self.model, 'temperature'):
            pass
        elif self.model.temperature != 1:
            raise ValueError(
                "Model temperature must be 1 for the flow models for SDDR computation."
            )
        
        allowed_models = ['HistogramModel', 'RealNVPModel', 'RQSplineModel']
        if self.model.__class__.__name__ not in allowed_models:
            warnings.warn(f"Model {self.model.__class__.__name__} is not supported for SDDR computation. Proceed at your own peril.")
        
        self.nsamples = self.samples.shape[0]
     
    def log_bayes_factor(self, 
                         log_prior: float, 
                         value: float | np.ndarray[tuple[np.float64]], 
                         bootstrap: bool = True,
                         nbootstraps: int = 10, 
                         bootstrap_proportion: float = 0.5, 
                         **kwargs) -> tuple:
        """Compute the log SDDR for the given value and log prior.

        Args:
            log_prior (float): Log prior value.
            value (float | np.ndarray): Value to compute the log Bayes factor via SDDR for.
            bootstrap (bool): Whether to use bootstrapping or not. 
                Bootstrapping is used to obtain errors on the SDDR estimate.
                Each bootstrapped sample is a random subset of the original samples.
                Those subsets are used to train independent models, which are then used to compute the SDDR.
            nbootstraps (int): Number of bootstraps to use.
            bootstrap_proportion (float): Proportion of samples to use for
                bootstrapping.
            **kwargs: Additional arguments to pass to the model.
        Returns:
            dict: Dictionary containing the log Bayes factor and its standard deviation.
        """
        
        if isinstance(value, float):
            value = np.asarray(value)
        
        if bootstrap:
            self.log_bf_bootstrapped = np.zeros(nbootstraps)
            for i in tqdm(range(nbootstraps), desc="Bootstrapping: "):
                bootstrapped_indices = np.random.choice(
                    self.nsamples,
                    int(self.nsamples * bootstrap_proportion),
                )
                bootstrapped_samples = self.samples[bootstrapped_indices, :]
                sddr_model = deepcopy(self.model)
                sddr_model.fit(X = bootstrapped_samples, **kwargs)
                 
                marginal_post_log_prob = sddr_model.predict(value)
                
                self.log_bf_bootstrapped[i] = marginal_post_log_prob - log_prior
                
                del sddr_model

            self.log_bf = self.log_bf_bootstrapped.mean()
            self.log_bf_std = self.log_bf_bootstrapped.std()
        else:
            sddr_model = deepcopy(self.model)
            sddr_model.fit(self.samples, **kwargs)
            self.log_bf = sddr_model.predict(value) - log_prior
            self.log_bf_std = None
            del sddr_model
            
        return (self.log_bf, self.log_bf_std)
    
    def bayes_factor(self,
                     prior: float,
                     value: float | np.ndarray[tuple[np.float64]],
                     bootstrap: bool = True,
                     nbootstraps: int = 10,
                     bootstrap_proportion: float = 0.5,
                     **kwargs) -> tuple:
        """Compute the log SDDR for the given value and log prior.

        Args:
            prior (float): Prior probability.
            value (float | np.ndarray): Value to compute the Bayes factor via SDDR for.
            bootstrap (bool): Whether to use bootstrapping or not. 
                Bootstrapping is used to obtain errors on the SDDR estimate.
                Each bootstrapped sample is a random subset of the original samples.
                Those subsets are used to train independent models, which are then used to compute the SDDR.
            nbootstraps (int): Number of bootstraps to use.
            bootstrap_proportion (float): Proportion of samples to use for
                bootstrapping.
            **kwargs: Additional arguments to pass to the model.
        Returns:
            dict: Dictionary containing the Bayes factor and its standard deviation.
        """
        
        _ = self.log_bayes_factor(np.log(prior), value, bootstrap, nbootstraps, bootstrap_proportion, **kwargs)
        
        self.bf = np.exp(self.log_bf)
        self.bf_std = np.exp(self.log_bf_bootstrapped).std()
        
        return (self.bf, self.bf_std)
            