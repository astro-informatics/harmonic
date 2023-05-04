from typing import Sequence, Callable
import model as md
import pickle
import numpy as np
import flows
import jax
import jax.numpy as jnp
import optax
from functools import partial
import tqdm
import flowMC

from flowMC.nfmodel.utils import make_training_loop
from flowMC.nfmodel.rqSpline import RQSpline
import flax
from flax.training import train_state  # Useful dataclass to keep train state
import distrax
import flax.linen as nn


class NormalizingFlow(md.Model):
    """Normalizing flow model to approximate the log_e posterior by a normalizing flow."""

    def __init__(
        self,
        ndim_in,
        flow=flows.NVPFlowLogProb(),
        scaling=0.8,
        opt=optax.adam(learning_rate=0.001),
        hyper_parameters=None,
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

            hyper_parameters (list): Hyper-parameters for model.

        Raises:

            ValueError: If the ndim_in is not positive.
            ValueError: If scaling is negative or greater than 1.

        """

        if ndim_in < 1:
            raise ValueError("Dimension must be greater than 0.")

        if scaling > 1:
            raise ValueError("Scaling must not be greater than 1.")

        if scaling <= 0:
            raise ValueError("Scaling must be positive.")

        self.ndim = ndim_in
        self.flow = flow
        self.fitted = False
        self.scaling = scaling
        self.optimizer = opt
        self.params = None

    def loss_fn(self, params, x):
        return -jnp.mean(self.flow.apply(params, x))

    @partial(jax.jit, static_argnums=(0,))
    def update_flow(self, params, opt_state, x):
        # Computes the gradients of the flow
        loss, grads = jax.value_and_grad(self.loss_fn)(params, x)

        # Computes the weights updates and apply them
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    def fit(self, X, Y, batch_size=5000, epochs=500, key=jax.random.PRNGKey(1000)):
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

        # Initializes the weights of the model
        params = self.flow.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
        opt_state = self.optimizer.init(params)

        np.random.shuffle(X)  # randomise samples order in chains
        batches_num = X.shape[0] / batch_size
        steps = int(epochs * batches_num)
        print("Running for ", steps, " steps.")
        batches = jnp.split(X, batches_num)

        for i in tqdm.tqdm(range(steps)):
            key, subkey = jax.random.split(key)

            # Get a batch of data
            x = batches[i % len(batches)]

            # Apply the update function
            params, opt_state, loss = self.update_flow(params, opt_state, x)

        self.params = params
        self.fitted = True

        return

    def predict(self, x):
        """Predict the value of the posterior at point x.

        Must be implemented by derived class (since abstract).

        Args:

            x (double ndarray[ndim]): Sample of shape (ndim) at which to
                predict posterior value.

        Returns:

            (double): Predicted log_e posterior value.

        """

        logprob = self.flow.apply(self.params, x, scale=self.scaling)

        return logprob

    def is_fitted(self):
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
        pickle.dump(self, file)
        file.close()

        return

    def deserialize(self, filename):
        """Deserialize Model object from file.

        Args:

            filename (string): Name of file from which to read model object.

        Returns:

            (Model): Model object deserialized from file.

        """
        file = open(filename, "rb")
        model = pickle.load(file)
        file.close()

        return model


class RQSplineFlow(md.Model):
    """Normalizing flow model to approximate the log_e posterior by a normalizing flow."""

    def __init__(
        self,
        ndim_in,
        scaling=0.8,
        n_layers=8,
        n_hiddens=[64, 64],
        n_bins=8,
        learning_rate=0.001,
        momentum=0.9,
        hyper_parameters=None,
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

            hyper_parameters (list): Hyper-parameters for model.

        Raises:

            ValueError: If the ndim_in is not positive.
            ValueError: If scaling is negative or greater than 1.

        """

        if ndim_in < 1:
            raise ValueError("Dimension must be greater than 0.")

        if scaling > 1:
            raise ValueError("Scaling must not be greater than 1.")

        if scaling <= 0:
            raise ValueError("Scaling must be positive.")

        self.ndim = ndim_in
        self.fitted = False
        self.scaling = scaling
        self.state = None

        # Model parameters
        self.n_layers = n_layers
        self.n_hiddens = n_hiddens
        self.n_bins = n_bins
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.flow = RQSpline(ndim_in, n_layers, n_hiddens, n_bins)

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

        key, rng_model, rng_init, rng_train, rng_nf_sample = jax.random.split(key, 5)

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

    def predict(self, x, compressed=True):
        """Predict the value of the posterior at point x.

        Must be implemented by derived class (since abstract).

        Args:

            x (double ndarray[ndim]): Sample of shape (ndim) at which to
                predict posterior value.

        Returns:

            (double): Predicted log_e posterior value.

        """

        if compressed:
            scaling = self.scaling
        if not compressed:
            scaling = 1.0

        logprob = self.flow.apply(
            {"params": self.state.params, "variables": self.variables},
            x,
            scaling,
            method=self.flow.log_prob_scaled,
        )

        return logprob

    def sample(self, n_sample, rng_key=jax.random.PRNGKey(0), compressed=True):
        """Sample n_sample samples from trained flow."""

        rng_key, subkey = jax.random.split(rng_key)

        if compressed:
            scaling = self.scaling
        if not compressed:
            scaling = 1.0

        samples = self.flow.apply(
            {"params": self.state.params, "variables": self.variables},
            subkey,
            n_sample,
            scaling,
            method=self.flow.sample,
        )

        return samples

    def is_fitted(self):
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
        pickle.dump(self, file)
        file.close()

        return

    def deserialize(self, filename):
        """Deserialize Model object from file.

        Args:

            filename (string): Name of file from which to read model object.

        Returns:

            (Model): Model object deserialized from file.

        """
        file = open(filename, "rb")
        model = pickle.load(file)
        file.close()

        return model


class RealNVP(md.Model):
    """Normalizing flow model to approximate the log_e posterior by a normalizing flow."""

    def __init__(
        self,
        ndim_in,
        hidden_layers=[128, 128],
        scaling=0.8,
        learning_rate=0.001,
        momentum=0.9,
        hyper_parameters=None,
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

            hyper_parameters (list): Hyper-parameters for model.

        Raises:

            ValueError: If the ndim_in is not positive.
            ValueError: If scaling is negative or greater than 1.

        """

        if ndim_in < 1:
            raise ValueError("Dimension must be greater than 0.")

        if scaling > 1:
            raise ValueError("Scaling must not be greater than 1.")

        if scaling <= 0:
            raise ValueError("Scaling must be positive.")

        self.ndim = ndim_in
        self.fitted = False
        self.scaling = scaling
        self.state = None

        # Model parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.flow = flows.RealNVP(ndim_in, hidden_layers)

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

        key, rng_model, rng_init, rng_train, rng_nf_sample = jax.random.split(key, 5)

        init = self.flow.init(rng_model, jnp.ones((1, self.ndim)))
        print(init.keys())
        variables = init

        state = self.create_train_state(rng_init)

        train_flow, train_epoch, train_step = make_training_loop(self.flow)
        rng, state, loss_values = train_flow(
            rng_train, state, variables, X, epochs, batch_size
        )

        self.state = state
        self.variables = variables
        self.fitted = True

        return

    def predict(self, x, compressed=True):
        """Predict the value of the posterior at point x.

        Must be implemented by derived class (since abstract).

        Args:

            x (double ndarray[ndim]): Sample of shape (ndim) at which to
                predict posterior value.

        Returns:

            (double): Predicted log_e posterior value.

        """

        if compressed:
            scaling = self.scaling
        if not compressed:
            scaling = 1.0

        logprob = self.flow.apply(
            {"params": self.state.params, "variables": self.variables},
            x,
            scaling,
            method=self.flow.log_prob,
        )

        return logprob

    def sample(self, n_sample, rng_key=jax.random.PRNGKey(0), compressed=True):
        """Sample n_sample samples from trained flow."""

        rng_key, subkey = jax.random.split(rng_key)

        if compressed:
            scaling = self.scaling
        if not compressed:
            scaling = 1.0

        samples = self.flow.apply(
            {"params": self.state.params, "variables": self.variables},
            subkey,
            n_sample,
            scaling,
            method=self.flow.sample,
        )

        return samples

    def is_fitted(self):
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
        pickle.dump(self, file)
        file.close()

        return

    def deserialize(self, filename):
        """Deserialize Model object from file.

        Args:

            filename (string): Name of file from which to read model object.

        Returns:

            (Model): Model object deserialized from file.

        """
        file = open(filename, "rb")
        model = pickle.load(file)
        file.close()

        return model
