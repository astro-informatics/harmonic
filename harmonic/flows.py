from typing import Sequence
import numpy as np
import flax.linen as nn
import optax
import pickle
import tqdm
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
import flowMC
from flax.training import train_state
from tqdm import trange

tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


# ===============================================================================
# NVP Flow
# ===============================================================================


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


def make_nvp_fn(d=2, scale=1.0):
    affine1 = AffineCoupling(name="affine1")
    affine2 = AffineCoupling(name="affine2")
    affine3 = AffineCoupling(name="affine3", apply_scaling=False)
    affine4 = AffineCoupling(name="affine4", apply_scaling=False)
    affine5 = AffineCoupling(name="affine5", apply_scaling=False)
    affine6 = AffineCoupling(name="affine6", apply_scaling=False)

    # Computes the likelihood of these x
    chain = tfb.Chain(
        [
            tfb.RealNVP(d // 2, bijector_fn=affine1),
            tfb.Permute([1, 0]),
            tfb.RealNVP(d // 2, bijector_fn=affine2),
            tfb.Permute([1, 0]),
            tfb.RealNVP(d // 2, bijector_fn=affine3),
            tfb.Permute([1, 0]),
            tfb.RealNVP(d // 2, bijector_fn=affine4),
            tfb.Permute([1, 0]),
            tfb.RealNVP(d // 2, bijector_fn=affine5),
            tfb.Permute([1, 0]),
            tfb.RealNVP(d // 2, bijector_fn=affine6),
        ]
    )

    nvp = tfd.TransformedDistribution(
        tfd.MultivariateNormalDiag(loc=jnp.zeros(2), scale_identity_multiplier=scale),
        bijector=chain,
    )
    return nvp


class NVPFlowLogProb(nn.Module):
    @nn.compact
    def __call__(self, x, scale=1.0):
        nvp = make_nvp_fn(scale=scale)
        return nvp.log_prob(x)


class NVPFlowSampler(nn.Module):
    @nn.compact
    def __call__(self, key, n_samples, scale=1.0):
        nvp = make_nvp_fn(scale=scale)
        return nvp.sample(n_samples, seed=key)


class RealNVP(nn.Module):
    """
    Real NVP flow using flax and tfp-jax.
    """

    n_features: int
    hidden_layers: Sequence[int]
    frac_masked: float = 0.5

    def setup(self):
        self.affine1 = AffineCoupling(name="affine1")
        self.affine2 = AffineCoupling(name="affine2")
        self.affine3 = AffineCoupling(name="affine3", apply_scaling=False)
        self.affine4 = AffineCoupling(name="affine4", apply_scaling=False)
        self.affine5 = AffineCoupling(name="affine5", apply_scaling=False)
        self.affine6 = AffineCoupling(name="affine6", apply_scaling=False)

    def make_flow(self, var_scale=1.0):

        """affine_coupling = tfb.real_nvp_default_template(
            hidden_layers=self.hidden_layers, activation=nn.leaky_relu
        )
        affine_coupling_unscaled = tfb.real_nvp_default_template(
            hidden_layers=self.hidden_layers, activation=nn.leaky_relu, shift_only=True
        )"""

        # Computes the likelihood of these x
        chain = tfb.Chain(
            [
                tfb.RealNVP(fraction_masked=self.frac_masked, bijector_fn=self.affine1),
                tfb.Permute([1, 0]),
                tfb.RealNVP(fraction_masked=self.frac_masked, bijector_fn=self.affine2),
                tfb.Permute([1, 0]),
                tfb.RealNVP(
                    fraction_masked=self.frac_masked,
                    bijector_fn=self.affine3,
                ),
                tfb.Permute([1, 0]),
                tfb.RealNVP(
                    fraction_masked=self.frac_masked,
                    bijector_fn=self.affine4,
                ),
                tfb.Permute([1, 0]),
                tfb.RealNVP(
                    fraction_masked=self.frac_masked,
                    bijector_fn=self.affine5,
                ),
                tfb.Permute([1, 0]),
                tfb.RealNVP(
                    fraction_masked=self.frac_masked,
                    bijector_fn=self.affine6,
                ),
            ]
        )

        nvp = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(
                loc=jnp.zeros(self.n_features), scale_identity_multiplier=var_scale
            ),
            bijector=chain,
        )

        return nvp

    def __call__(self, x, var_scale=1.0) -> jnp.array:
        # x = (x - self.base_mean.value) / jnp.sqrt(jnp.diag(self.base_cov.value))
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
