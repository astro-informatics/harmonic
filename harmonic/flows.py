from typing import Sequence, Callable, List, Any
import numpy as np
import flax.linen as nn
import optax
import pickle
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
from flax.training import train_state

tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


class AffineCoupling(nn.Module):
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
    
class ShiftLogScale(nn.Module):
  hidden_layers: List[int]
  shift_only: bool = False
  activation: Callable[..., Any] = jax.nn.relu

  @nn.compact
  def __call__(self, x, output_units):
    if x.ndim == 1:
      x = x[jnp.newaxis, ...]
      reshape_output = lambda x: x[0]
    else:
      reshape_output = lambda x: x
    for units in self.hidden_layers:
      x = self.activation(nn.Dense(units)(x))
    x = nn.Dense((1 if self.shift_only else 2) * output_units)(x)
    if self.shift_only:
      return reshape_output(x), None
    shift, log_scale = jnp.split(x, 2, axis=-1)
    return reshape_output(shift), reshape_output(log_scale)



# ===============================================================================
# NVP Flow
# ===============================================================================


class RealNVP(nn.Module):
    """
    Real NVP flow using flax and tfp-jax.
    """

    n_features: int
    # hidden_layers: Sequence[int]
    frac_masked: float = 0.5
    n_scaled_layers: int = 2
    n_unscaled_layers: int = 4

    def setup(self):
        self.scaled_layers = [AffineCoupling() for i in range(self.n_scaled_layers)]
        self.unscaled_layers = [
            AffineCoupling(apply_scaling=False) for i in range(self.n_unscaled_layers)
        ]
        self.base_mean = jnp.zeros(self.n_features)
        self.base_cov = jnp.eye(self.n_features)

    def make_flow(self, var_scale=1.0):
        chain = []
        ix = jnp.arange(self.n_features)
        permutation = [ix[-1], *ix[:-1]]

        # assume n_scaled_layers is not 0
        for i in range(self.n_scaled_layers - 1):
            chain.append(
                tfb.RealNVP(
                    fraction_masked=self.frac_masked, bijector_fn=self.scaled_layers[i]
                )
            )
            chain.append(tfb.Permute(permutation))

        chain.append(
            tfb.RealNVP(
                fraction_masked=self.frac_masked, bijector_fn=self.scaled_layers[-1]
            )
        )

        for i in range(self.n_unscaled_layers):
            chain.append(tfb.Permute(permutation))
            chain.append(
                tfb.RealNVP(
                    fraction_masked=self.frac_masked,
                    bijector_fn=self.unscaled_layers[i],
                )
            )

        # Computes the likelihood of these x
        chain = tfb.Chain(chain)

        nvp = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=jnp.zeros(self.n_features), scale_diag=jnp.full(self.n_features, var_scale)),
            bijector=chain,
        )

        return nvp

    def __call__(self, x, var_scale=1.0) -> jnp.array:
        #x = (x - self.base_mean) / jnp.sqrt(jnp.diag(self.base_cov))
        flow = self.make_flow(var_scale=var_scale)
        return flow.log_prob(x)

    def sample(
        self, rng: jax.random.PRNGKey, num_samples: int, var_scale: float = 1.0
    ) -> jnp.array:
        """ "
        Sample from the flow.
        """
        nvp = self.make_flow(var_scale=var_scale)
        samples = nvp.sample(num_samples, seed=rng)

        return samples

    def log_prob(self, x: jnp.array, var_scale: float = 1.0) -> jnp.array:
        get_logprob = jax.jit(jax.vmap(self.__call__, in_axes=[0, None]))
        logprob = get_logprob(x, var_scale)
        return logprob