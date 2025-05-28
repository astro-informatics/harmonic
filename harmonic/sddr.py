import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from copy import deepcopy
import warnings

@dataclass
class sddr:
    """Class to compute the Bayes factor between two models.

    Args:
        model (Model): The density estimator model to use.
        samples (np.ndarray): The samples to use for the Bayes factor computation.
    """
    model: object
    samples: np.ndarray

    def __post_init__(self):
        if self.model.ndim < 1:
            raise ValueError("ndim must be greater than 0.")
        if self.model.is_fitted():
            raise ValueError("Model is already fitted, pass in an unfitted model.")
        
        if self.model.ndim != self.samples.shape[1]:
            raise ValueError(
                f"Model ndim {self.model.ndim} does not match samples shape {self.samples.shape[1]}."
            )
        
        if not hasattr(self.model, 'temperature'):
            pass
        elif self.model.temperature == 1:
            pass
        else:
            raise ValueError(
                "Model temperature must be None or 1 for SDDR computation."
            )
        
        allowed_models = ['HistogramModel', 'RealNVPModel', 'RQSplineModel']
        if self.model.__class__.__name__ not in allowed_models:
            warnings.warn(f"Model {self.model.__class__.__name__} is not supported for SDDR computation. Proceed at your own peril.")
        
        self.nsamples = self.samples.shape[0]
     
    def log_bayes_factor(self, 
                         log_prior: float, 
                         value: float | np.ndarray, 
                         nbootstraps: int = 10, 
                         bootstrap_proportion: float = 0.5, 
                         bootstrap: bool = True,
                         **kwargs) -> dict:
        """Compute the log SDDR for the given value and log prior.

        Args:
            log_prior (float): Log prior value.
            value (float): Value to compute the log Bayes factor via SDDR for.
            nbootstraps (int): Number of bootstraps to use.
            bootstrap_proportion (float): Proportion of samples to use for
                bootstrapping.
            bootstrap (bool): Whether to use bootstrapping or not.
            **kwargs: Additional arguments to pass to the model.
        Returns:
            dict: Dictionary containing the log Bayes factor and its standard deviation.
        """
        
        if isinstance(value, float):
            value = np.asarray(value)
        
        if bootstrap:
            self.log_bf_bootstrapped = np.zeros(nbootstraps)
            for i in tqdm(range(nbootstraps)):
                bootstrapped_indices = np.random.choice(
                    self.nsamples,
                    int(self.nsamples * bootstrap_proportion),
                )
                bootstrapped_samples = self.samples[bootstrapped_indices, :]
                sddr_model = deepcopy(self.model)
                sddr_model.fit(X = bootstrapped_samples, **kwargs)
                 
                if isinstance(value, int):  # To account for the 1D case
                    value = np.array(value)
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
            
        return {'log_bf': self.log_bf, 'log_bf_std': self.log_bf_std}
    
    def bayes_factor(self,
                     prior: float,
                     value: float | np.ndarray,
                     nbootstraps: int = 10,
                     bootstrap_proportion: float = 0.5,
                     bootstrap: bool = True,
                     **kwargs) -> dict:
        """Compute the log SDDR for the given value and log prior.

        Args:
            prior (float): Prior probability.
            value (float): Value to compute the Bayes factor via SDDR for.
            nbootstraps (int): Number of bootstraps to use.
            bootstrap_proportion (float): Proportion of samples to use for
                bootstrapping.
            bootstrap (bool): Whether to use bootstrapping or not.
            **kwargs: Additional arguments to pass to the model.
        Returns:
            dict: Dictionary containing the Bayes factor and its standard deviation.
        """
        
        _ = self.log_bayes_factor(np.log(prior), value, nbootstraps, bootstrap_proportion, bootstrap, **kwargs)
        
        if self.log_bf:
            self.bf = np.exp(self.log_bf)
        if self.log_bf_std:
            self.bf_std = np.exp(self.log_bf_bootstrapped).std()
        
        return {'bf': self.bf, 'bf_std': self.bf_std}
            