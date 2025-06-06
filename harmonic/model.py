from typing import Sequence
from harmonic import model_abstract as mda
from harmonic import flows
import jax
import jax.numpy as jnp
import optax
from tqdm import trange
from flax.training import train_state  # Useful dataclass to keep train state


def make_training_loop(model):
    """
    Create a function that trains an NF model.

    Args:
        model: a neural network model with a `log_prob` method.

    Returns:
        train_flow (Callable): wrapper function that trains the model.

    Note:
        Adapted from github.com/kazewong/flowMC
    """

    def train_step(batch, state, variables):
        def loss(params):
            log_det = model.apply(
                {"params": params, "variables": variables},
                batch,
                temperature=1.0,
                method=model.log_prob,
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

    def train_flow(
        rng,
        state,
        variables,
        data: jnp.ndarray,
        num_epochs: int,
        batch_size: int,
        verbose: bool = False,
    ):
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
            loss_values = loss_values.at[epoch].set(value)
            if loss_values[epoch] < best_loss:
                best_state = state
                best_loss = loss_values[epoch]
            if verbose:
                if num_epochs > 10:
                    if epoch % int(num_epochs / 10) == 0:
                        pbar.set_description(f"Training NF")
                else:
                    if epoch == num_epochs:
                        pbar.set_description(f"Training NF")

        return rng, best_state, loss_values

    return train_flow, train_epoch, train_step


class FlowModel(mda.Model):
    """Normalizing flow model to approximate the log_e posterior by a normalizing flow."""

    def __init__(
        self,
        ndim_in: int,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        standardize: bool = False,
        temperature: float = 0.8,
    ):
        if ndim_in < 1:
            raise ValueError("Dimension must be greater than 0.")

        self.ndim = ndim_in
        self.fitted = False
        self.state = None

        # Model parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.standardize = standardize
        self.temperature = temperature
        self.flow = None

    def create_train_state(self, rng):
        params = self.flow.init(rng, jnp.ones((1, self.ndim)))["params"]
        tx = optax.adam(self.learning_rate, self.momentum)
        return train_state.TrainState.create(
            apply_fn=self.flow.apply, params=params, tx=tx
        )

    def fit(
        self,
        X: jnp.ndarray,
        batch_size: int = 64,
        epochs: int = 3,
        key=jax.random.PRNGKey(1000),
        verbose: bool = False,
    ):
        """Fit the parameters of the model.

        Args:

            X (jnp.ndarray (nsamples, ndim)): Training samples.

            batch_size (int, optional): Batch size used when training flow. Default = 64.

            epochs (int, optional): Number of epochs flow is trained for. Default = 3.

            key (Union[jax.Array, jax.random.PRNGKeyArray], optional): Key used in random number generation process.

            verbose (bool, optional): Controls if progress bar and current loss are displayed when training. Default = False.


        Raises:

            ValueError: Raised if the second dimension of X is not the same as ndim.

            NotImplementedError: If called directly from FlowModel class.

        """

        if self.flow is None:
            raise NotImplementedError(
                "This method cannot be used in the FlowModel class directly. Use a class with a specific flow implemented (RealNVPModel, RQSplineModel)."
            )

        if X.shape[1] != self.ndim:
            raise ValueError("X second dimension not the same as ndim.")

        key, rng_model, rng_init, rng_train = jax.random.split(key, 4)

        variables = self.flow.init(rng_model, jnp.ones((1, self.ndim)))
        state = self.create_train_state(rng_init)

        # set up standardisation
        if self.standardize:
            # self.pre_offset = jnp.min(X, axis = 0) #maxmin
            self.pre_offset = jnp.mean(X, axis=0)
            # self.pre_amp = (jnp.max(X, axis=0) - self.pre_offset)
            if X.shape[1] > 1:
                self.pre_amp = jnp.sqrt(jnp.diag(jnp.cov(X.T)))
            else:
                self.pre_amp = jnp.sqrt(jnp.std(X, axis=0))

            X = (X - self.pre_offset) / self.pre_amp

        train_flow, train_epoch, train_step = make_training_loop(self.flow)
        rng, state, loss_values = train_flow(
            rng_train, state, variables, X, epochs, batch_size, verbose=verbose
        )

        self.state = state
        self.variables = variables
        self.fitted = True

        return

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """Predict the value of log_e posterior at batched input x.

        Args:

            x (jnp.ndarray (batch_size, ndim)): Sample for which to
                predict posterior values.

        Returns:

            jnp.ndarray (batch_size,): Predicted log_e posterior value.

        Raises:

            ValueError: If temperature is negative or greater than 1.

        """

        temperature = self.temperature

        if temperature > 1:
            raise ValueError("Scaling must not be greater than 1.")

        if temperature <= 0:
            raise ValueError("Scaling must be positive.")

        if self.standardize:
            x = (x - self.pre_offset) / self.pre_amp

        logprob = self.flow.apply(
            {"params": self.state.params, "variables": self.variables},
            x,
            temperature,
            method=None
            if len(x.shape) == 1
            else self.flow.log_prob,  # 1D input must be handled by directly calling the flow
        )

        if self.standardize:
            logprob -= sum(jnp.log(self.pre_amp))

        return logprob

    def sample(self, n_sample: int, rng_key=jax.random.PRNGKey(0)) -> jnp.ndarray:
        """Sample from trained flow.

        Args:
            nsample (int): Number of samples generated.

            rng_key (Union[jax.Array, jax.random.PRNGKeyArray]), optional): Key used in random number generation process.

        Raises:

            ValueError: If temperature is negative or greater than 1.

        Returns:

            jnp.array (n_sample, ndim): Samples from fitted distribution.
        """

        temperature = self.temperature

        if temperature > 1:
            raise ValueError("Scaling must not be greater than 1.")

        if temperature <= 0:
            raise ValueError("Scaling must be positive.")

        samples = self.flow.apply(
            {"params": self.state.params, "variables": self.variables},
            rng_key,
            n_sample,
            temperature,
            method=self.flow.sample,
        )

        if self.standardize:
            samples = (samples * self.pre_amp) + self.pre_offset

        return samples


# ===============================================================================
# NVP Flow
# ===============================================================================


class RealNVPModel(FlowModel):
    """Normalizing flow model to approximate the log_e posterior by a NVP normalizing flow."""

    def __init__(
        self,
        ndim_in: int,
        n_scaled_layers: int = 2,
        n_unscaled_layers: int = 4,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        standardize: bool = False,
        temperature: float = 0.8,
    ):
        """Constructor setting the hyper-parameters of the model.

        Args:

            ndim_in (int): Dimension of the problem to solve.

            n_scaled_layers (int, optional): Number of layers with scaler in RealNVP flow. Default = 2.

            n_unscaled_layers (int, optional): Number of layers without scaler in RealNVP flow. Default = 4.

            learning_rate (float, optional): Learning rate for adam optimizer used in the fit method. Default = 0.001.

            momentum (float, optional): Learning rate for Adam optimizer used in the fit method. Default = 0.9

            standardize(bool, optional): Indicates if mean and variance should be removed from training data when training the flow. Default = False

            temperature (float, optional): Scale factor by which the base distribution Gaussian is compressed in the prediction step. Should be positive and <=1. Default = 0.8.

        Raises:

            ValueError: If the ndim_in is not positive.

            ValueError: If n_scaled_layers is not positive.

        """

        if n_scaled_layers <= 0:
            raise ValueError("Number of scaled layers must be greater than 0.")

        FlowModel.__init__(
            self,
            ndim_in,
            learning_rate,
            momentum,
            standardize,
            temperature,
        )

        # Model parameters
        self.n_scaled_layers = n_scaled_layers
        self.n_unscaled_layers = n_unscaled_layers
        self.flow = flows.RealNVP(ndim_in, self.n_scaled_layers, self.n_unscaled_layers)


# ===============================================================================
# Rational Quadratic Spline Flow
# ===============================================================================


class RQSplineModel(FlowModel):
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
        temperature: float = 0.8,
    ):
        """Constructor setting the hyper-parameters and domains of the model.

        Must be implemented by derived class (currently abstract).

        Args:

            ndim_in (int): Dimension of the problem to solve.

            n_layers (int, optional): Number of layers in the flow. Defaults to 8.

            n_bins (int, optional): Number of bins in the spline. Defaults to 8.

            hidden_size (Sequence[int], optional): Size of the hidden layers in the conditioner. Defaults to [64, 64].

            spline_range (Sequence[float], optional): Range of the spline. Defaults to (-10.0, 10.0).

            standardize (bool, optional): Indicates if mean and variance should be removed from training data when training the flow. Defaults to False.

            learning_rate (float, optional): Learning rate for adam optimizer used in the fit method. Defaults to 0.001.

            momentum (float, optional): Learning rate for Adam optimizer used in the fit method. Defaults to 0.9.

            temperature (float, optional): Scale factor by which the base distribution Gaussian is compressed in the prediction step. Should be positive and <=1. Defaults to 0.8.

        Raises:

            ValueError: If the ndim_in is not positive.

        """

        FlowModel.__init__(
            self,
            ndim_in,
            learning_rate,
            momentum,
            standardize,
            temperature,
        )

        # Flow parameters
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_bins = n_bins
        self.spline_range = spline_range
        self.flow = flows.RQSpline(ndim_in, n_layers, hidden_size, n_bins, spline_range)
