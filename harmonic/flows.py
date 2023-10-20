from typing import Sequence, Callable
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
import distrax

tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


# ===============================================================================
# NVP Flow
# ===============================================================================


class RealNVP(nn.Module):
    """
    Real-valued non-volume preserving flow using flax and tfp-jax.

    Args:
        n_features (int): Number of features in the data.

        n_scaled_layers (int, optional): Non-zero number of layers in the flow. Defaults to 2.

        n_unscaled_layers (int, optional): Number of unscaled layers in the flow. Defaults to 4.

    """

    n_features: int
    n_scaled_layers: int = 2
    n_unscaled_layers: int = 4

    def setup(self):
        self.scaled_layers = [AffineCoupling() for i in range(self.n_scaled_layers)]
        self.unscaled_layers = [
            AffineCoupling(apply_scaling=False) for i in range(self.n_unscaled_layers)
        ]

    def make_flow(self, var_scale: float = 1.0):
        """
        Make tfp-jax distribution object containing the RealNVP flow.

        Args:
            var_scale (float, optional): Factor by which base Gaussian unit covariance matrix is scaled.
                Should be between 0 and 1 for use in evidence estimation. Defaults to 1.


        Returns:
            tfb.Distribution: Base Gaussian transformed by scaled contained in the scaled_layers
                attribute, followed by unscaled affine coupling layers contained in the
                unscaled_layers attribute.

        Raises:

            ValueError: If n_scaled_layers is not positive.

        """

        if self.n_scaled_layers <= 0:
            raise ValueError("Number of scaled layers must be greater than zero.")

        chain = []
        ix = jnp.arange(self.n_features)
        permutation = [ix[-1], *ix[:-1]]

        # assume n_scaled_layers is not 0
        for i in range(self.n_scaled_layers - 1):
            chain.append(
                tfb.RealNVP(fraction_masked=0.5, bijector_fn=self.scaled_layers[i])
            )
            chain.append(tfb.Permute(permutation))

        chain.append(
            tfb.RealNVP(fraction_masked=0.5, bijector_fn=self.scaled_layers[-1])
        )

        for i in range(self.n_unscaled_layers):
            chain.append(tfb.Permute(permutation))
            chain.append(
                tfb.RealNVP(
                    fraction_masked=0.5,
                    bijector_fn=self.unscaled_layers[i],
                )
            )

        # Computes the likelihood of these x
        chain = tfb.Chain(chain)

        nvp = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(
                loc=jnp.zeros(self.n_features),
                scale_diag=jnp.full(self.n_features, var_scale),
            ),
            bijector=chain,
        )

        return nvp

    def __call__(self, x: jnp.ndarray, var_scale: int = 1.0) -> jnp.array:
        """
        Evaluate the log probability of the flow for non-batched input x.

        Args:
            x (jnp.ndarray (ndim)): Sample at which to predict posterior value.

            var_scale (float, optional): Factor by which base Gaussian unit covariance matrix is scaled.
                Should be between 0 and 1 for use in evidence estimation. Defaults to 1.

        Returns:
            float: Predicted log_e posterior value.
        """

        flow = self.make_flow(var_scale=var_scale)
        return flow.log_prob(x)

    def sample(
        self, rng: jax.random.PRNGKey, num_samples: int, var_scale: float = 1.0
    ) -> jnp.array:
        """ "
        Sample from the flow.

        Args:
            rng (Union[Array, PRNGKeyArray])): Key used in random number generation process.

            num_samples (int): Number of samples generated.

            var_scale (float, optional): Factor by which base Gaussian unit covariance matrix is scaled.
                Should be between 0 and 1 for use in evidence estimation. Defaults to 1.

        Returns:
            jnp.array (num_samples, ndim): Samples from fitted distribution.
        """
        nvp = self.make_flow(var_scale=var_scale)
        samples = nvp.sample(num_samples, seed=rng)

        return samples

    def log_prob(self, x: jnp.array, var_scale: float = 1.0) -> jnp.array:
        """
        Evaluate the log probability of the flow for a batched input.

        Args:
            x (jnp.ndarray (batch_size, ndim)): Sample for which to predict posterior values.

            var_scale (float, optional): Factor by which base Gaussian unit covariance matrix is scaled.
                Should be between 0 and 1 for use in evidence estimation. Defaults to 1.

        Returns:
            jnp.ndarray (batch_size,): Predicted log_e posterior value.
        """
        get_logprob = jax.jit(jax.vmap(self.__call__, in_axes=[0, None]))
        logprob = get_logprob(x, var_scale)

        return logprob


class AffineCoupling(nn.Module):
    """
    Affine coupling layer used in RealNVP flow class.

    Args:
        apply_scaling (bool, optional): If true shift is followed by a scaling. Defaults to True.
    """

    apply_scaling: bool = True

    @nn.compact
    def __call__(self, x, nunits):
        net = nn.leaky_relu(nn.Dense(128)(x))

        # Shift parameter:
        shift = nn.Dense(nunits)(net)

        if self.apply_scaling:
            scaler = tfb.Scale(jnp.clip(nn.softplus(nn.Dense(nunits)(net)), 1e-3, 1e3))
        else:
            scaler = tfb.Identity()
        return tfb.Chain([tfb.Shift(shift), scaler])


# ===============================================================================
# RQSpline Flow
# ===============================================================================


class RQSpline(nn.Module):
    """
    Rational quadratic spline normalizing flow model using distrax.

    Args:
        n_features (int): Number of features in the data.

        num_layers (int): Number of layers in the flow.

        num_bins (int): Number of bins in the spline.

        hidden_size (Sequence[int]): Size of the hidden layers in the conditioner.

        spline_range (Sequence[float], optional): Range of the spline. Defaults to (-10, 10)

    Note:
        Adapted from github.com/kazewong/flowMC
    """

    n_features: int
    num_layers: int
    hidden_size: Sequence[int]
    num_bins: int
    spline_range: Sequence[float] = (-10.0, 10.0)

    def setup(self):
        conditioner = []
        scalar = []
        for i in range(self.num_layers):
            conditioner.append(
                Conditioner(self.n_features, self.hidden_size, 3 * self.num_bins + 1)
            )
            scalar.append(Scalar(self.n_features))

        self.conditioner = conditioner
        self.scalar = scalar

        self.vmap_call = jax.jit(jax.vmap(self.__call__))

        def bijector_fn(params: jnp.ndarray):
            return distrax.RationalQuadraticSpline(
                params, range_min=self.spline_range[0], range_max=self.spline_range[1]
            )

        self.bijector_fn = bijector_fn

    def make_flow(self, var_scale: float = 1.0):
        """
        Make distrax distribution containing the rational quadratic spline flow.

        Args:
            var_scale (float, optional): Factor by which base Gaussian unit covariance matrix is scaled.
                Should be between 0 and 1 for use in evidence estimation. Defaults to 1.

        Returns:
            Base Gaussian transformed by rational quadratic spline flow.
        """
        mask = (jnp.arange(0, self.n_features) % 2).astype(bool)
        mask_all = (jnp.zeros(self.n_features)).astype(bool)
        layers = []
        for i in range(self.num_layers):
            layers.append(
                distrax.MaskedCoupling(
                    mask=mask_all, bijector=scalar_affine, conditioner=self.scalar[i]
                )
            )
            layers.append(
                distrax.MaskedCoupling(
                    mask=mask,
                    bijector=self.bijector_fn,
                    conditioner=self.conditioner[i],
                )
            )
            mask = jnp.logical_not(mask)

        flow = distrax.Inverse(distrax.Chain(layers))
        base_dist = distrax.Independent(
            distrax.MultivariateNormalFullCovariance(
                loc=jnp.zeros(self.n_features),
                covariance_matrix=jnp.eye(self.n_features) * var_scale,
            )
        )

        return base_dist, flow

    def __call__(self, x: jnp.array, var_scale: float = 1.0) -> jnp.array:
        """
        Evaluate the log probability of the flow for non-batched input x.

        Args:
            x (jnp.ndarray (ndim)): Sample at which to predict posterior value.

            var_scale (float, optional): Factor by which base Gaussian unit covariance matrix is scaled.
                Should be between 0 and 1 for use in evidence estimation. Defaults to 1.

        Returns:
            jnp.ndarray (float): Predicted log_e posterior value.
        """
        base_dist, flow = self.make_flow(var_scale=var_scale)

        return distrax.Transformed(base_dist, flow).log_prob(x)

    def sample(
        self, rng: jax.random.PRNGKey, num_samples: int, var_scale: float = 1.0
    ) -> jnp.array:
        """ "
        Sample from the flow.

        Args:
            rng (Union[Array, PRNGKeyArray])): Key used in random number generation process.

            num_samples (int): Number of samples generated.

            var_scale (float, optional): Factor by which base Gaussian unit covariance matrix is scaled.
                Should be between 0 and 1 for use in evidence estimation. Defaults to 1.

        Returns:
            jnp.array (num_samples, ndim): Samples from fitted distribution.
        """

        base_dist, flow = self.make_flow(var_scale=var_scale)
        samples = distrax.Transformed(base_dist, flow).sample(
            seed=rng, sample_shape=(num_samples)
        )

        return samples

    def log_prob(self, x: jnp.array, var_scale: float = 1.0) -> jnp.array:
        """
        Evaluate the log probability of the flow for a batched input.

        Args:
            x (jnp.ndarray (batch_size, ndim)): Sample for which to predict posterior values.

            var_scale (float, optional): Factor by which base Gaussian unit covariance matrix is scaled.
                Should be between 0 and 1 for use in evidence estimation. Defaults to 1.

        Returns:
            jnp.ndarray (batch_size,): Predicted log_e posterior value.
        """

        get_logprob = jax.jit(jax.vmap(self.__call__, in_axes=[0, None]))
        logprob = get_logprob(x, var_scale)

        return logprob


class Reshape(nn.Module):
    shape: Sequence[int]

    def __call__(self, x: jnp.ndarray):
        return jnp.reshape(x.T, self.shape)


class Conditioner(nn.Module):
    """
    Conditioner used to construct the bijector function in RQSpline class.

    Adapted from github.com/kazewong/flowMC
    """

    n_features: int
    hidden_size: Sequence[int]
    num_bijector_params: int

    def setup(self):
        self.conditioner = nn.Sequential(
            [
                MLP(
                    [self.n_features] + list(self.hidden_size),
                    nn.tanh,
                    init_weight_scale=1e-2,
                ),
                nn.Dense(
                    self.n_features * self.num_bijector_params,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                ),
                Reshape((self.n_features, self.num_bijector_params)),
            ]
        )

    def __call__(self, x):
        return self.conditioner(x)


class Scalar(nn.Module):
    """
    Scalar used to construct the spline flow in RQSpline class.

    Adapted from github.com/kazewong/flowMC
    """

    n_features: int

    def setup(self):
        self.shift = self.param(
            "shifts", lambda rng, shape: jnp.zeros(shape), (self.n_features)
        )
        self.scale = self.param(
            "scales", lambda rng, shape: jnp.ones(shape), (self.n_features)
        )

    def __call__(self, x):
        return self.scale, self.shift


def scalar_affine(params: jnp.ndarray):
    return distrax.ScalarAffine(scale=params[0], shift=params[1])


class MLP(nn.Module):
    """
    Multi-layer perceptron in Flax. We use a gaussian kernel with a standard deviation
    of `init_weight_scale=1e-4` by default.

    Args:
        features (list of int): The number of features in each layer.

        activation (callable, optional): The activation function at each level Defaults to nn.relu.

        use_bias (bool, optional): Whether to use bias in the layers. Defaults to True.

        init_weight_scale (float, optional): The initial weight scale for the layers. Defaults to 1e-4.

        kernel_init (callable, optional): The kernel initializer for the layers. Defaults to jax.nn.initializers.variance_scaling.

    Adapted from github.com/kazewong/flowMC
    """

    features: Sequence[int]
    activation: Callable = nn.relu
    use_bias: bool = True
    init_weight_scale: float = 1e-4
    kernel_i: Callable = jax.nn.initializers.variance_scaling

    def setup(self):
        self.layers = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"),
            )
            for feat in self.features
        ]

    def __call__(self, x):
        for l, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
