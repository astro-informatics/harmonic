import numpy as np
import flax.linen as nn
import optax
import pickle
import tqdm
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp

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
