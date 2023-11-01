import numpy as np
import sys
import emcee
import time
import matplotlib.pyplot as plt
from functools import partial

sys.path.append(".")
import harmonic as hm

sys.path.append("examples")
import utils
from harmonic import model_nf


def ln_prior_uniform(x, xmin=-10.0, xmax=10.0, ymin=-5.0, ymax=15.0):
    """Compute log_e of uniform prior.

    Args:

        x: Position at which to evaluate prior.

        xmin: Uniform prior minimum x edge (first dimension).

        xmax: Uniform prior maximum x edge (first dimension).

        ymin: Uniform prior minimum y edge (second dimension).

        ymax: Uniform prior maximum y edge (second dimension).

    Returns:

        double: Value of prior at specified point.

    """

    if x[0] >= xmin and x[0] <= xmax and x[1] >= ymin and x[1] <= ymax:
        return 1.0 / ((xmax - xmin) * (ymax - ymin))
    else:
        return 0.0


def ln_prior_gaussian(x, mu=1.0, sigma=5.0):
    """Compute log_e of Gaussian prior.

    Args:

        x: Position at which to evaluate prior.

        mu: Mean (centre) of the prior.

        sigma: Standard deviation of prior.

    Returns:

        double: Value of prior at specified point.

    """

    return -0.5 * np.dot(x - mu, x - mu) / sigma**2 - 0.5 * x.size * np.log(
        2 * np.pi * sigma
    )


def ln_likelihood(x, a=1.0, b=100.0):
    """Compute log_e of likelihood defined by Rosenbrock function.

    Args:

        x: Position at which to evaluate likelihood.

        a: First parameter of Rosenbrock function.

        b: First parameter of Rosenbrock function.

    Returns:

        double: Value of Rosenbrock at specified point.

    """

    ndim = x.size

    f = 0.0

    for i_dim in range(ndim - 1):
        f += b * (x[i_dim + 1] - x[i_dim] ** 2) ** 2 + (a - x[i_dim]) ** 2

    return -f


def ln_posterior(x, ln_prior, a=1.0, b=100.0):
    """Compute log_e of posterior.

    Args:

        x: Position at which to evaluate posterior.

        a: First parameter of Rosenbrock function.

        b: First parameter of Rosenbrock function.

        ln_prior: Prior function.

    Returns:

        double: Posterior at specified point.

    """

    ln_L = ln_likelihood(x, a=a, b=b)

    if not np.isfinite(ln_L):
        return -np.inf
    else:
        return ln_prior(x) + ln_L


def run_example(
    ndim=2, nchains=100, samples_per_chain=1000, nburn=500, plot_corner=False
):
    """Run Rosenbrock example.

    Args:

        ndim: Dimension.

        nchains: Number of chains.

        samples_per_chain: Number of samples per chain.

        nburn: Number of burn in samples for each chain.

        plot_corner: Plot marginalised distributions if true.
    """

    if ndim != 2:
        raise ValueError("Only ndim=2 is supported (ndim={} specified)".format(ndim))

    # ===========================================================================
    # Configure Parameters.
    # ===========================================================================
    """
    Configure machine learning parameters
    """
    savefigs = True
    a = 1.0
    b = 100.0
    epochs_num = 5
    temperature = 0.8
    training_proportion = 0.5
    standardize = True
    """
    Set prior parameters.
    """
    use_uniform_prior = True
    if use_uniform_prior:
        xmin = -10.0
        xmax = 10.0
        ymin = -5.0
        ymax = 15.0
        hm.logs.debug_log(
            "xmin, xmax, ymin, ymax = {}, {}, {}, {}".format(xmin, xmax, ymin, ymax)
        )
        ln_prior = partial(ln_prior_uniform, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    else:  # Use Gaussian prior
        mu = 1.0
        sigma = 50.0
        hm.logs.debug_log("a, b, mu, sigma = {}, {}, {}, {}".format(a, b, mu, sigma))
        ln_prior = partial(ln_prior_gaussian, mu=mu, sigma=sigma)

    # Start timer.
    clock = time.process_time()

    # ===========================================================================
    # Begin multiple realisations of estimator
    # ===========================================================================
    """
    Set up and run multiple simulations
    """
    n_realisations = 1
    evidence_inv_summary = np.zeros((n_realisations, 3))
    for i_realisation in range(n_realisations):
        if n_realisations > 1:
            hm.logs.info_log(
                "Realisation number = {}/{}".format(i_realisation + 1, n_realisations)
            )

        # =======================================================================
        # Run Emcee to recover posterior samples
        # =======================================================================
        hm.logs.info_log("Run sampling...")
        """
        Feed emcee the ln_posterior function, starting positions and recover 
        chains.
        """
        pos = np.random.rand(ndim * nchains).reshape((nchains, ndim)) * 0.1
        sampler = emcee.EnsembleSampler(
            nchains, ndim, ln_posterior, args=[ln_prior, a, b]
        )
        rstate = np.random.get_state()
        sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
        samples = np.ascontiguousarray(sampler.chain[:, nburn:, :])
        lnprob = np.ascontiguousarray(sampler.lnprobability[:, nburn:])

        # =======================================================================
        # Configure emcee chains for harmonic
        # =======================================================================
        hm.logs.info_log("Configure chains...")
        """
        Configure chains for the cross-validation stage.
        """
        chains = hm.Chains(ndim)
        chains.add_chains_3d(samples, lnprob)
        chains_train, chains_test = hm.utils.split_data(
            chains, training_proportion=training_proportion
        )

        # =======================================================================
        # Fit model
        # =======================================================================
        hm.logs.info_log("Fit model for {} epochs...".format(epochs_num))
        model = model_nf.RQSplineModel(
            ndim, standardize=standardize, temperature=temperature
        )
        model.fit(chains_train.samples, epochs=epochs_num)

        # =======================================================================
        # Computing evidence using learnt model and emcee chains
        # =======================================================================
        hm.logs.info_log("Compute evidence...")
        """
        Instantiates the evidence class with a given model. Adds some chains and 
        computes the log-space evidence (marginal likelihood).
        """
        ev = hm.Evidence(chains_test.nchains, model)
        ev.add_chains(chains_test)
        ln_evidence, ln_evidence_std = ev.compute_ln_evidence()

        # Compute analytic evidence.
        if ndim == 2:
            hm.logs.debug_log("Compute evidence by numerical integration...")
            ln_posterior_func = partial(ln_posterior, ln_prior=ln_prior, a=a, b=b)
            ln_posterior_grid, x_grid, y_grid = utils.eval_func_on_grid(
                ln_posterior_func,
                xmin=-10.0,
                xmax=10.0,
                ymin=-5.0,
                ymax=15.0,
                nx=1000,
                ny=1000,
            )
            dx = x_grid[0, 1] - x_grid[0, 0]
            dy = y_grid[1, 0] - y_grid[0, 0]
            evidence_numerical_integration = np.sum(np.exp(ln_posterior_grid)) * dx * dy

        # ======================================================================
        # Display evidence computation results.
        # ======================================================================
        hm.logs.debug_log(
            "Evidence: numerical = {}, estimate = {}".format(
                evidence_numerical_integration, np.exp(ln_evidence)
            )
        )
        hm.logs.debug_log(
            "Evidence: std = {}, std / estimate = {}".format(
                np.exp(ln_evidence_std), np.exp(ln_evidence_std - ln_evidence)
            )
        )
        diff = np.log(np.abs(evidence_numerical_integration - np.exp(ln_evidence)))
        hm.logs.info_log(
            "Evidence: |numerical - estimate| / estimate = {}".format(
                np.exp(diff - ln_evidence)
            )
        )

        # ======================================================================
        # Display inverse evidence computation results.
        # ======================================================================
        hm.logs.debug_log(
            "Inv Evidence: numerical = {}, estimate = {}".format(
                1.0 / evidence_numerical_integration, ev.evidence_inv
            )
        )
        hm.logs.debug_log(
            "Inv Evidence: std = {}, std / estimate = {}".format(
                np.sqrt(ev.evidence_inv_var),
                np.sqrt(ev.evidence_inv_var) / ev.evidence_inv,
            )
        )
        hm.logs.debug_log(
            "Inv Evidence: kurtosis = {}, sqrt( 2 / ( n_eff - 1 ) ) = {}".format(
                ev.kurtosis, np.sqrt(2.0 / (ev.n_eff - 1))
            )
        )
        hm.logs.debug_log(
            "Inv Evidence: sqrt( var(var) )/ var = {}".format(
                np.sqrt(ev.evidence_inv_var_var) / ev.evidence_inv_var
            )
        )
        hm.logs.info_log(
            "Inv Evidence: |numerical - estimate| / estimate = {}".format(
                np.abs(1.0 / evidence_numerical_integration - ev.evidence_inv)
                / ev.evidence_inv
            )
        )

        # ===========================================================================
        # Display more technical details
        # ===========================================================================
        hm.logs.debug_log("---------------------------------")
        hm.logs.debug_log("Technical Details")
        hm.logs.debug_log("---------------------------------")
        hm.logs.debug_log(
            "lnargmax = {}, lnargmin = {}".format(ev.lnargmax, ev.lnargmin)
        )
        hm.logs.debug_log(
            "lnprobmax = {}, lnprobmin = {}".format(ev.lnprobmax, ev.lnprobmin)
        )
        hm.logs.debug_log(
            "lnpredictmax = {}, lnpredictmin = {}".format(
                ev.lnpredictmax, ev.lnpredictmin
            )
        )
        hm.logs.debug_log("---------------------------------")
        hm.logs.debug_log(
            "shift = {}, shift setting = {}".format(ev.shift_value, ev.shift)
        )
        hm.logs.debug_log("running sum total = {}".format(sum(ev.running_sum)))
        hm.logs.debug_log("running sum = \n{}".format(ev.running_sum))
        hm.logs.debug_log("nsamples per chain = \n{}".format(ev.nsamples_per_chain))
        hm.logs.debug_log(
            "nsamples eff per chain = \n{}".format(ev.nsamples_eff_per_chain)
        )
        hm.logs.debug_log("===============================")

        # Create corner/triangle plot.
        created_plots = False
        if plot_corner and i_realisation == 0:
            utils.plot_getdist(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig(
                    "examples/plots/spline_rosenbrock_getdist.png", bbox_inches="tight"
                )

            plt.show(block=False)

            # =======================================================================
            # Visualise distributions
            # =======================================================================

            num_samp = chains_train.samples.shape[0]
            samps_compressed = np.array(model.sample(num_samp))

            utils.plot_getdist_compare(chains_train.samples, samps_compressed)
            if savefigs:
                plt.savefig(
                    "examples/plots/spline_rosenbrock_corner_all_T"
                    + str(temperature)
                    + ".png",
                    bbox_inches="tight",
                    dpi=300,
                )
            created_plots = True

        # Save out realisations for voilin plot.
        evidence_inv_summary[i_realisation, 0] = ev.evidence_inv
        evidence_inv_summary[i_realisation, 1] = ev.evidence_inv_var
        evidence_inv_summary[i_realisation, 2] = ev.evidence_inv_var_var

    # ===========================================================================
    # End Timer.
    clock = time.process_time() - clock
    hm.logs.info_log("Execution time = {}s".format(clock))

    # ===========================================================================
    # Save out realisations of statistics for analysis.
    if n_realisations > 1:
        np.savetxt(
            "examples/data/spline_rosenbrock_evidence_inv_T"
            + str(temperature)
            + "_realisations.dat",
            evidence_inv_summary,
        )
        evidence_inv_analytic_summary = np.zeros(1)
        evidence_inv_analytic_summary[0] = 1.0 / evidence_numerical_integration
        np.savetxt(
            "examples/data/spline_rosenbrock_evidence_inv" + "_analytic.dat",
            evidence_inv_analytic_summary,
        )

    if created_plots:
        input("\nPress Enter to continue...")

    return samples


if __name__ == "__main__":
    # Setup logging config.
    hm.logs.setup_logging()

    # Define parameters.
    ndim = 2
    nchains = 200
    samples_per_chain = 5000
    nburn = 2000
    np.random.seed(20)

    hm.logs.info_log("Rosenbrock example")

    hm.logs.debug_log("-- Selected Parameters --")

    hm.logs.debug_log("Dimensionality = {}".format(ndim))
    hm.logs.debug_log("Number of chains = {}".format(nchains))
    hm.logs.debug_log("Samples per chain = {}".format(samples_per_chain))
    hm.logs.debug_log("Burn in = {}".format(nburn))

    hm.logs.debug_log("-------------------------")

    # Run example.
    samples = run_example(ndim, nchains, samples_per_chain, nburn, plot_corner=True)
