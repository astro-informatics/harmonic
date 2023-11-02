import jax.numpy as jnp
import jax
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import numpy as np
from getdist import plots, MCSamples
import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors
from harmonic import model as md
from harmonic import flows


def get_moon(sigma, resolution=1024):
    """
    Generate a TFP approximate distribution of the two moons dataset
    Parameters
    ----------
    sigma: float
    Spread of the 2 moons distribution.
    resolution: int
    Number of components in the gaussian mixture approximation of the
    distribution (default: 1024)
    Returns
    -------
    distribution: TFP distribution
    Two moons distribution
    """

    outer_circ_x = jnp.cos(jnp.linspace(0, jnp.pi, resolution))
    outer_circ_y = jnp.sin(jnp.linspace(0, jnp.pi, resolution))
    # inner_circ_x = 1 - jnp.cos(jnp.linspace(0, jnp.pi, resolution))
    # inner_circ_y = 1 - jnp.sin(jnp.linspace(0, jnp.pi, resolution)) - .5

    X = outer_circ_x
    Y = outer_circ_y
    coords = jnp.vstack([X, Y])
    # coords = X

    distribution = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=jnp.ones(resolution) / resolution),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=coords.T, scale_diag=jnp.full(coords.shape[0], sigma)
        ),
    )
    return distribution


def get_two_moons(sigma, resolution=1024):
    """
    Generate a TFP approximate distribution of the two moons dataset
    Parameters
    ----------
    sigma: float
      Spread of the 2 moons distribution.
    resolution: int
      Number of components in the gaussian mixture approximation of the
      distribution (default: 1024)
    Returns
    -------
    distribution: TFP distribution
      Two moons distribution
    """

    outer_circ_x = jnp.cos(jnp.linspace(0, jnp.pi, resolution))
    outer_circ_y = jnp.sin(jnp.linspace(0, jnp.pi, resolution))
    inner_circ_x = 1 - jnp.cos(jnp.linspace(0, jnp.pi, resolution))
    inner_circ_y = 1 - jnp.sin(jnp.linspace(0, jnp.pi, resolution)) - 0.5

    X = jnp.append(outer_circ_x, inner_circ_x)
    Y = jnp.append(outer_circ_y, inner_circ_y)
    coords = jnp.vstack([X, Y])

    distribution = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=jnp.ones(2 * resolution) / resolution / 2
        ),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=coords.T, scale_diag=jnp.full(coords.shape[0], sigma)
        ),
    )
    return distribution


@jax.jit
def get_batch(seed):
    return get_moon(0.05).sample(batch_size, seed=seed)


if __name__ == "__main__":
    # Let's build a function that can sample from a two-moons distribution
    batch_size = 20000
    seed = jax.random.PRNGKey(42)

    ndim = 2
    n_scaled = 4
    n_unscaled = 2
    epochs_num = 100

    posterior_samples = np.array(get_moon(0.05).sample(batch_size, seed=seed))
    model = md.RealNVPModel(
        ndim, n_scaled_layers=n_scaled, n_unscaled_layers=n_unscaled
    )
    model.fit(posterior_samples, epochs=epochs_num, verbose=True)

    num_samp = batch_size * 5

    # samps = np.array(model.sample(num_samp, temperature=1.))
    model.temperature = 0.7
    samps2 = np.array(model.sample(num_samp))
    model.temperature = 0.4
    samps3 = np.array(model.sample(num_samp))

    # Get the getdist MCSamples objects for the samples, specifying same parameter
    # names and labels; if not specified weights are assumed to all be unity
    names = ["x%s" % i for i in range(ndim)]
    labels = ["x_%s" % i for i in range(ndim)]
    samples_train = MCSamples(
        samples=posterior_samples, names=names, labels=labels, label="Posterior"
    )
    # samples = MCSamples(samples=samps,names = names, labels = labels, label = 'T = 1.0')
    samples2 = MCSamples(samples=samps2, names=names, labels=labels, label="T = 0.7")
    samples3 = MCSamples(samples=samps3, names=names, labels=labels, label="T = 0.4")

    # 2D line contour comparison plot with extra bands and markers
    g = plots.get_single_plotter()
    g.settings.linewidth = 1
    g.plot_2d(
        [samples_train, samples2, samples3],
        "x0",
        "x1",
        colors=["orange", "green", "#800080"],
        filled=True,
    )
    # colors=[('#F31212', '#F68A8A'), ('#17DB1E', '#83DB86'), ('#0080FF', '#89C4FF')])
    g.add_legend(["Posterior", "T=0.7", "T=0.4"], colored_text=True)
    plt.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )
    plt.xlim(-1.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.savefig("examples/plots/temperature_diag.png", bbox_inches="tight", dpi=300)
    plt.show()
