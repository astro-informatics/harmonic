from typing import Sequence, Callable, List
from harmonic import model as md
import pickle
import numpy as np
from harmonic import flows
import jax
import jax.numpy as jnp
import optax
from functools import partial
import tqdm
import flowMC
from examples.utils import plot_getdist_compare
import matplotlib.pyplot as plt

from flowMC.nfmodel.utils import make_training_loop
from flowMC.nfmodel.rqSpline import RQSpline
import flax
from flax.training import train_state  # Useful dataclass to keep train state
import distrax
import flax.linen as nn


# ===============================================================================
# Rational Quadratic Spline Flow - to be refactored (cf RealNVP)
# ===============================================================================


class RQSplineFlow(md.Model):
    """Rational quadratic spline flow model to approximate the log_e posterior by a normalizing flow."""

    def __init__(
        self,
        ndim_in,
        n_layers=8,
        n_hiddens=[64, 64],
        n_bins=8,
        learning_rate=0.001,
        momentum=0.9,
    ):
        """Constructor setting the hyper-parameters and domains of the model.

        Must be implemented by derived class (currently abstract).

        Args:

            ndim (int): Dimension of the problem to solve.

            domains (list): List of 1D numpy ndarrays containing the domains
                for each parameter of model. Each domain is of length two,
                specifying a lower and upper bound for real hyper-parameters
                but can be different in other cases if required.

            flow (Flow): Normalizing flow used to approximate the posterior.

            scaling (float): Scale factor by which the base distribution Gaussian
                is compressed in the prediction step. Should be positive and <=1.

            learning_rate (float): Learning rate for adam optimizer used in the fit method.

            momentum (float): Learning rate for Adam optimizer used in the fit method.

        Raises:

            ValueError: If the ndim_in is not positive.
            ValueError: If scaling is negative or greater than 1.

        """

        if ndim_in < 1:
            raise ValueError("Dimension must be greater than 0.")

        self.ndim = ndim_in
        self.fitted = False
        self.state = None

        # Model parameters
        self.n_layers = n_layers
        self.n_hiddens = n_hiddens
        self.n_bins = n_bins
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.flow = RQSpline(ndim_in, n_layers, n_hiddens, n_bins)

    def is_fitted(self):
        """Specify whether model has been fitted.

        Returns:

            (bool): Whether the model has been fitted.

        """

        return self.fitted

    def create_train_state(self, rng):
        params = self.flow.init(rng, jnp.ones((1, self.ndim)))["params"]
        tx = optax.adam(self.learning_rate, self.momentum)
        return train_state.TrainState.create(
            apply_fn=self.flow.apply, params=params, tx=tx
        )

    def fit(self, X, Y, batch_size=10000, epochs=3, key=jax.random.PRNGKey(1000)):
        """Fit the parameters of the model.

        Args:

            X (double ndarray[nsamples, ndim]): Sample x coordinates.

            Y (double ndarray[nsamples]): Target log_e posterior values for each
                sample in X.

        Returns:

            (bool): Whether fit successful.


        Raises:

            ValueError: Raised if the first dimension of X is not the same as
                Y.

            ValueError: Raised if the second dimension of X is not the same as
                ndim.

        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y sizes are not the same.")

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim.")

        key, rng_model, rng_init, rng_train = jax.random.split(key, 4)

        variables = self.flow.init(rng_model, jnp.ones((1, self.ndim)))["variables"]
        variables = variables.unfreeze()
        variables["base_mean"] = jnp.mean(X, axis=0)
        variables["base_cov"] = jnp.cov(X.T)
        variables = flax.core.freeze(variables)

        state = self.create_train_state(rng_init)

        train_flow, train_epoch, train_step = make_training_loop(self.flow)
        rng, state, loss_values = train_flow(
            rng_train, state, variables, X, epochs, batch_size
        )

        self.state = state
        self.variables = variables
        self.fitted = True

        return

    def predict(self, x, var_scale: float = 1.0):
        """Predict the value of log_e posterior at x.

        Args:

            x (jnp.ndarray): Sample of shape at which to
                predict posterior value.

            var_scale (float): Scale factor by which the base distribution Gaussian
                is compressed in the prediction step. Should be positive and <=1.

        Returns:

            jnp.ndarray: Predicted log_e posterior value.

        """

        if var_scale > 1:
            raise ValueError("Scaling must not be greater than 1.")

        if var_scale <= 0:
            raise ValueError("Scaling must be positive.")

        logprob = self.flow.apply(
            {"params": self.state.params, "variables": self.variables},
            x,
            var_scale,
            method=self.flow.log_prob_scaled,
        )

        return logprob

    def sample(self, n_sample, rng_key=jax.random.PRNGKey(0), var_scale: float = 1.0):
        """Sample from trained flow.

        Args:
            nsample (int): Number of samples generated.

            rng_key (Union[Array, PRNGKeyArray])): Key used in random number generation process.

            var_scale (float): Scale factor by which the base distribution Gaussian
                is compressed in the prediction step. Should be positive and <=1.

        Returns:

            jnp.array (n_sample, ndim): Samples from fitted distribution."""

        if var_scale > 1:
            raise ValueError("Scaling must not be greater than 1.")

        if var_scale <= 0:
            raise ValueError("Scaling must be positive.")

        samples = self.flow.apply(
            {"params": self.state.params, "variables": self.variables},
            rng_key,
            n_sample,
            var_scale,
            method=self.flow.sample,
        )

        return samples


# ===============================================================================
# NVP Flow - will generalise this to take a custom flow
# ===============================================================================


class RealNVPModel(md.Model):
    """Normalizing flow model to approximate the log_e posterior by a NVP normalizing flow."""

    def __init__(
        self,
        ndim_in,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        flow = None
    ):
        """Constructor setting the hyper-parameters of the model.

        Args:

            ndim_in (int): Dimension of the problem to solve.

            learning_rate (float): Learning rate for adam optimizer used in the fit method.

            momentum (float): Learning rate for Adam optimizer used in the fit method.

        Raises:

            ValueError: If the ndim_in is not positive.
            ValueError: If scaling is negative or greater than 1.

        """

        if ndim_in < 1:
            raise ValueError("Dimension must be greater than 0.")

        self.ndim = ndim_in
        self.fitted = False
        self.state = None

        # Model parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        if flow is None:
            self.flow = flows.RealNVP(ndim_in)
        else:
            self.flow = flow
        self.pre_offset = jnp.zeros(ndim_in)
        self.pre_amp = jnp.ones(ndim_in)

    def is_fitted(self):
        """Specify whether model has been fitted.

        Returns:

            (bool): Whether the model has been fitted.

        """

        return self.fitted

    def create_train_state(self, rng):
        params = self.flow.init(rng, jnp.ones((1, self.ndim)))["params"]
        tx = optax.adam(self.learning_rate, self.momentum)
        return train_state.TrainState.create(
            apply_fn=self.flow.apply, params=params, tx=tx
        )


    def fit(self, X, Y, batch_size=10000, epochs=3, key=jax.random.PRNGKey(1000), standardize = False):
        """Fit the parameters of the model.

        Args:

            X (double ndarray[nsamples, ndim]): Sample x coordinates.

            Y (double ndarray[nsamples]): Target log_e posterior values for each
                sample in X.


        Raises:

            ValueError: Raised if the first dimension of X is not the same as
                Y.

            ValueError: Raised if the second dimension of X is not the same as
                ndim.

        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y sizes are not the same.")

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim.")

        key, rng_model, rng_init, rng_train = jax.random.split(key, 4)

        variables = self.flow.init(rng_model, jnp.ones((1, self.ndim)))
        state = self.create_train_state(rng_init)

        #set up standardisation
        if standardize:
            self.pre_offset = jnp.min(X, axis = 0)
            self.pre_amp = (jnp.max(X, axis=0) - self.pre_offset)

        X_old = X
        X = (X - self.pre_offset) / self.pre_amp
        print("max", jnp.max(X, axis=0), "min", jnp.min(X, axis = 0), "amp", self.pre_amp)
              
        plot_getdist_compare(X_old, X)
        plt.show()

        train_flow, train_epoch, train_step = make_training_loop(self.flow)
        rng, state, loss_values = train_flow(
            rng_train, state, variables, X, epochs, batch_size
        )

        self.state = state
        self.variables = variables
        self.fitted = True

        return

    def predict(self, x, var_scale: float = 1.0):
        """Predict the value of log_e posterior at x.

        Args:

            x (jnp.ndarray): Sample of shape at which to
                predict posterior value.

            var_scale (float): Scale factor by which the base distribution Gaussian
                is compressed in the prediction step. Should be positive and <=1.

        Returns:

            jnp.ndarray: Predicted log_e posterior value.

        """

        if var_scale > 1:
            raise ValueError("Scaling must not be greater than 1.")

        if var_scale <= 0:
            raise ValueError("Scaling must be positive.")
        
        x = (x-self.pre_offset)/self.pre_amp
        print("predict max", jnp.max(x, axis=0), "min", jnp.min(x, axis = 0))

        logprob = self.flow.apply(
            {"params": self.state.params, "variables": self.variables},
            x,
            var_scale,
            method=self.flow.log_prob,
        )

        logprob -= sum(jnp.log(self.pre_amp))

        return logprob

    def sample(self, n_sample, rng_key=jax.random.PRNGKey(0), var_scale: float = 1.0):
        """Sample from trained flow.

        Args:
            nsample (int): Number of samples generated.

            rng_key (Union[Array, PRNGKeyArray])): Key used in random number generation process.

            var_scale (float): Scale factor by which the base distribution Gaussian
                is compressed in the prediction step. Should be positive and <=1.

        Returns:

            jnp.array (n_sample, ndim): Samples from fitted distribution.
        """

        if var_scale > 1:
            raise ValueError("Scaling must not be greater than 1.")

        if var_scale <= 0:
            raise ValueError("Scaling must be positive.")

        samples = self.flow.apply(
            {"params": self.state.params, "variables": self.variables},
            rng_key,
            n_sample,
            var_scale,
            method=self.flow.sample,
        )

        #samples = (samples * jnp.sqrt(jnp.diag(self.base_cov))) + self.base_mean
        samples = (samples * self.pre_amp) + self.pre_offset
        return samples
