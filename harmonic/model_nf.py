from typing import Sequence, Callable, List
from harmonic import model as md
import numpy as np
from harmonic import flows
import jax
import jax.numpy as jnp
import optax
from functools import partial
from examples.utils import plot_getdist_compare
import matplotlib.pyplot as plt
from tqdm import trange

import flax
from flax.training import train_state  # Useful dataclass to keep train state
import flax.linen as nn

def make_training_loop(model):
    """
    Create a function that trains an NF model.

    Args:
        model: a neural network model with a `log_prob` method.

    Returns:
        train_flow (Callable): wrapper function that trains the model.
    """

    def train_step(batch, state, variables):
        def loss(params):
            log_det = model.apply(
                {"params": params, "variables": variables}, batch, method=model.log_prob
            )
            return -jnp.mean(log_det)

        grad_fn = jax.value_and_grad(loss)
        value, grad = grad_fn(state.params)
        state = state.apply_gradients(grads=grad)
        return value, state

    train_step = jax.jit(train_step)

    def train_epoch(rng, state, variables, train_ds, batch_size):
        """Train for a single epoch."""
        train_ds_size = len(train_ds)
        steps_per_epoch = train_ds_size // batch_size
        if steps_per_epoch > 0:
            perms = jax.random.permutation(rng, train_ds_size)

            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            for perm in perms:
                batch = train_ds[perm, ...]
                value, state = train_step(batch, state, variables)
        else:
            value, state = train_step(train_ds, state, variables)

        return value, state

    def train_flow(rng, state, variables, data, num_epochs, batch_size, verbose: bool = False):
        loss_values = jnp.zeros(num_epochs)
        if verbose:
            pbar = trange(num_epochs, desc="Training NF", miniters=int(num_epochs / 10))
        else:
            pbar = range(num_epochs)
        best_state = state
        best_loss = 1e9
        for epoch in pbar:
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            value, state = train_epoch(input_rng, state, variables, data, batch_size)
            # print('Train loss: %.3f' % value)
            loss_values = loss_values.at[epoch].set(value)
            if loss_values[epoch] < best_loss:
                best_state = state
                best_loss = loss_values[epoch]
            if verbose:
                if num_epochs > 10:
                    if epoch % int(num_epochs / 10) == 0:
                        pbar.set_description(f"Training NF, current loss: {value:.3f}")
                else:
                    if epoch == num_epochs:
                        pbar.set_description(f"Training NF, current loss: {value:.3f}")

        return rng, best_state, loss_values

    return train_flow, train_epoch, train_step



# ===============================================================================
# NVP Flow - will generalise this to take a custom flow
# ===============================================================================


class RealNVPModel(md.Model):
    """Normalizing flow model to approximate the log_e posterior by a NVP normalizing flow."""

    def __init__(
        self,
        ndim_in: int,
        n_scaled_layers: int = 2,
        n_unscaled_layers: int = 4,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        standardize: bool = False,
        temperature: float = 0.8
    ):
        """Constructor setting the hyper-parameters of the model.

        Args:

            ndim_in (int): Dimension of the problem to solve.

            n_scaled_layers (int): Number of layers with scaler in RealNVP flow.

            n_unscaled_layers (int): Number of layers without scaler in RealNVP flow.

            learning_rate (float): Learning rate for adam optimizer used in the fit method.

            momentum (float): Learning rate for Adam optimizer used in the fit method.

            standardize(bool): Indicates if mean and variance should be removed from training data when training the flow.

            temperature (float): Scale factor by which the base distribution Gaussian is compressed in the prediction step. Should be positive and <=1.

        Raises:

            ValueError: If the ndim_in is not positive.

        """

        if ndim_in < 1:
            raise ValueError("Dimension must be greater than 0.")

        self.ndim = ndim_in
        self.fitted = False
        self.state = None

        # Model parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_scaled_layers = n_scaled_layers
        self.n_unscaled_layers = n_unscaled_layers
        self.standardize = standardize
        self.flow = flows.RealNVP(ndim_in, self.n_scaled_layers, self.n_unscaled_layers)
        self.temperature = temperature

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


    def fit(self, X, batch_size=64, epochs=3, key=jax.random.PRNGKey(1000)):
        """Fit the parameters of the model.

        Args:

            X (double ndarray[nsamples, ndim]): Training samples.

            batch_size (int): Batch size used when training flow.

            epochs (int): Number of epochs flow is trained for.

            key (Union[Array, PRNGKeyArray])): Key used in random number generation process.


        Raises:

            ValueError: Raised if the second dimension of X is not the same as
                ndim.

        """

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim.")

        key, rng_model, rng_init, rng_train = jax.random.split(key, 4)

        variables = self.flow.init(rng_model, jnp.ones((1, self.ndim)))
        state = self.create_train_state(rng_init)

        #set up standardisation
        if self.standardize:
            #self.pre_offset = jnp.min(X, axis = 0) #maxmin
            self.pre_offset = jnp.mean(X, axis=0)
            #self.pre_amp = (jnp.max(X, axis=0) - self.pre_offset)
            self.pre_amp = jnp.sqrt(jnp.diag(jnp.cov(X.T)))

            X = (X - self.pre_offset) / self.pre_amp

        train_flow, train_epoch, train_step = make_training_loop(self.flow)
        rng, state, loss_values = train_flow(
            rng_train, state, variables, X, epochs, batch_size
        )

        self.state = state
        self.variables = variables
        self.fitted = True

        return

    def predict(self, x):
        """Predict the value of log_e posterior at x.

        Args:

            x (jnp.ndarray): Sample of shape at which to
                predict posterior value.

        Returns:

            jnp.ndarray: Predicted log_e posterior value.

        Raises:
        
            ValueError: If var_scale is negative or greater than 1.

        """

        var_scale = self.temperature

        if var_scale > 1:
            raise ValueError("Scaling must not be greater than 1.")

        if var_scale <= 0:
            raise ValueError("Scaling must be positive.")
        
        if self.standardize:
            x = (x-self.pre_offset)/self.pre_amp
            print("predict max", jnp.max(x, axis=0), "min", jnp.min(x, axis = 0))

        logprob = self.flow.apply(
            {"params": self.state.params, "variables": self.variables},
            x,
            var_scale,
            method=self.flow.log_prob,
        )

        if self.standardize:
            logprob -= sum(jnp.log(self.pre_amp))

        return logprob

    def sample(self, n_sample, rng_key=jax.random.PRNGKey(0)):
        """Sample from trained flow.

        Args:
            nsample (int): Number of samples generated.

            rng_key (Union[Array, PRNGKeyArray])): Key used in random number generation process.

        Returns:

            jnp.array (n_sample, ndim): Samples from fitted distribution.
        """

        var_scale = self.temperature

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
        
        if self.standardize:
            samples = (samples * self.pre_amp) + self.pre_offset
        
        return samples


# ===============================================================================
# Rational Quadratic Spline Flow - to be refactored (cf RealNVP)
# ===============================================================================


class RQSplineFlow(md.Model):
    """Rational quadratic spline flow model to approximate the log_e posterior by a normalizing flow."""

    def __init__(
        self,
        ndim_in: int,
        n_layers: int = 8,
        n_bins: int = 8,
        hidden_size: Sequence[int] = [64, 64],
        spline_range: Sequence[float] = (-10.0, 10.0),
        standardize: bool = False,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
    ):
        """Constructor setting the hyper-parameters and domains of the model.

        Must be implemented by derived class (currently abstract).

        Args:

            ndim_in (int): Dimension of the problem to solve.

            num_layers : (int) Number of layers in the flow.

            n_bins : (int) Number of bins in the spline.

            hidden_size : (Sequence[int]) Size of the hidden layers in the conditioner.

            spline_range : (Sequence[float]) Range of the spline.

            standardize(bool): Indicates if mean and variance should be removed from training data when training the flow.

            learning_rate (float): Learning rate for adam optimizer used in the fit method.

            momentum (float): Learning rate for Adam optimizer used in the fit method.

        Raises:

            ValueError: If the ndim_in is not positive.

        """

        if ndim_in < 1:
            raise ValueError("Dimension must be greater than 0.")

        self.ndim = ndim_in
        self.fitted = False
        self.state = None
        self.standardize = standardize

        # Flow parameters
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_bins = n_bins
        self.spline_range = spline_range
        self.flow = flows.RQSpline(ndim_in, n_layers, hidden_size, n_bins, spline_range)

        # Optimizer parameters
        self.learning_rate = learning_rate
        self.momentum = momentum

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

    def fit(self, X, batch_size=64, epochs=3, key=jax.random.PRNGKey(1000)):
        """Fit the parameters of the model.

        Args:

            X (double ndarray[nsamples, ndim]): Sample x coordinates.

            batch_size (int): Batch size used when training flow.

            epochs (int): Number of epochs flow is trained for.

            key (Union[Array, PRNGKeyArray])): Key used in random number generation process.


        Raises:

            ValueError: Raised if the second dimension of X is not the same as
                ndim.

        """

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim.")

        key, rng_model, rng_init, rng_train = jax.random.split(key, 4)

        variables = self.flow.init(rng_model, jnp.ones((1, self.ndim)))
        state = self.create_train_state(rng_init)

        #set up standardisation
        if self.standardize:
            #self.pre_offset = jnp.min(X, axis = 0) #maxmin
            self.pre_offset = jnp.mean(X, axis=0)
            #self.pre_amp = (jnp.max(X, axis=0) - self.pre_offset)
            self.pre_amp = jnp.sqrt(jnp.diag(jnp.cov(X.T)))

            X = (X - self.pre_offset) / self.pre_amp

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
        
        Raises:

            ValueError: If var_scale is negative or greater than 1.

        """

        if var_scale > 1:
            raise ValueError("Scaling must not be greater than 1.")

        if var_scale <= 0:
            raise ValueError("Scaling must be positive.")

        logprob = self.flow.apply(
            {"params": self.state.params, "variables": self.variables},
            x,
            var_scale,
            method=self.flow.log_prob,
        )

        if self.standardize:
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

        if self.standardize:
            samples = (samples * self.pre_amp) + self.pre_offset

        return samples
