import numpy as np
import time
import matplotlib.pyplot as plt
import harmonic as hm
import jax
import jax.numpy as jnp
import emcee


def ln_analytic_evidence(ndim, cov):
    """Compute analytic evidence for nD Gaussian.

    Args:

        ndim: Dimension of Gaussian.

        cov: Covariance matrix.

    Returns:

        double: Analytic evidence.

    """

    ln_norm_lik = 0.5 * ndim * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(cov))
    return ln_norm_lik


# @partial(jax.jit, static_argnums=(1,))
def ln_posterior(x, inv_cov):
    """Compute log_e of posterior.

    Args:

        x: Position at which to evaluate posterior.

        inv_cov: Inverse covariance matrix.

    Returns:

        double: Value of Gaussian at specified point.

    """

    return -jnp.dot(x, jnp.dot(inv_cov, x)) / 2.0


def init_cov(ndim):
    """Initialise random non-diagonal covariance matrix.

    Args:

        ndim: Dimension of Gaussian.

    Returns:

        cov: Covariance matrix of shape (ndim,ndim).

    """

    cov = np.zeros((ndim, ndim))
    diag_cov = np.ones(ndim) + np.random.randn(ndim) * 0.1
    np.fill_diagonal(cov, diag_cov)

    for i in range(ndim - 1):
        cov[i, i + 1] = (-1) ** i * 0.5 * np.sqrt(cov[i, i] * cov[i + 1, i + 1])
        cov[i + 1, i] = cov[i, i + 1]

    return cov


def run_example(
    flow_type,
    ndim=2,
    nchains=100,
    samples_per_chain=1000,
    plot_corner=False,
):
    """Run Gaussian example with non-diagonal covariance matrix.

    Args:

        flow_type: Which flow model to use, "RealNVP" or "RQSpline".

        ndim: Dimension.

        nchains: Number of chains.

        samples_per_chain: Number of samples per chain.

        plot_corner: Plot marginalised distributions if true.

    """

    savefigs = True

    # Initialise covariance matrix.
    cov = init_cov(ndim)
    inv_cov = jnp.linalg.inv(cov)
    training_proportion = 0.5
    if flow_type == "RealNVP":
        epochs_num = 5
    elif flow_type == "RQSpline":
        # epochs_num = 3
        epochs_num = 100

    # Beginning of path where plots will be saved
    save_name_start = "examples/plots/" + flow_type

    temperature = 0.8
    standardize = True
    verbose = True

    # Spline params
    n_layers = 5
    n_bins = 16
    hidden_size = [32, 32]
    spline_range = (-10.0, 10.0)

    # Start timer.
    clock = time.process_time()

    # Run multiple realisations.
    n_realisations = 1
    evidence_inv_summary = np.zeros((n_realisations, 3))
    for i_realisation in range(n_realisations):
        if n_realisations > 0:
            hm.logs.info_log(
                "Realisation = {}/{}".format(i_realisation + 1, n_realisations)
            )
        # Define the number of dimensions and the mean of the Gaussian
        num_samples = nchains * samples_per_chain
        # Initialize a PRNG key (you can use any valid key)
        key = jax.random.PRNGKey(0)
        mean = jnp.zeros(ndim)

        # Generate random samples from the 2D Gaussian distribution
        samples = jax.random.multivariate_normal(key, mean, cov, shape=(num_samples,))
        lnprob = jax.vmap(ln_posterior, in_axes=(0, None))(samples, jnp.array(inv_cov))
        samples = jnp.reshape(samples, (nchains, -1, ndim))
        lnprob = jnp.reshape(lnprob, (nchains, -1))

        MCMC = False
        if MCMC:
            nburn = 500
            # Set up and run sampler.
            hm.logs.info_log("Run sampling...")

            pos = np.random.rand(ndim * nchains).reshape((nchains, ndim))

            sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=[inv_cov])
            rstate = np.random.get_state()  # Set random state to repeatable
            # across calls.
            (pos, prob, state) = sampler.run_mcmc(
                pos, samples_per_chain, rstate0=rstate
            )
            samples = np.ascontiguousarray(sampler.chain[:, nburn:, :])
            lnprob = np.ascontiguousarray(sampler.lnprobability[:, nburn:])

        # Calculate evidence using harmonic....

        # Set up chains.
        chains = hm.Chains(ndim)
        chains.add_chains_3d(samples, lnprob)
        chains_train, chains_test = hm.utils.split_data(
            chains, training_proportion=training_proportion
        )

        # =======================================================================
        # Fit model
        # =======================================================================
        hm.logs.info_log("Fit model for {} epochs...".format(epochs_num))
        if flow_type == "RealNVP":
            model = hm.model.RealNVPModel(
                ndim, standardize=standardize, temperature=temperature
            )
        if flow_type == "RQSpline":
            model = hm.model.RQSplineModel(
                ndim,
                n_layers=n_layers,
                n_bins=n_bins,
                hidden_size=hidden_size,
                spline_range=spline_range,
                standardize=standardize,
                temperature=temperature,
            )
        model.fit(chains_train.samples, epochs=epochs_num, verbose=verbose)

        # Use chains and model to compute inverse evidence.
        hm.logs.info_log("Compute evidence...")

        ev = hm.Evidence(chains_test.nchains, model)
        # ev.set_mean_shift(0.0)
        ev.add_chains(chains_test)
        err_ln_inv_evidence = ev.compute_ln_inv_evidence_errors()

        # Compute analytic evidence.
        if i_realisation == 0:
            ln_evidence_analytic = ln_analytic_evidence(ndim, cov)

        hm.logs.info_log("---------------------------------")
        hm.logs.info_log("The inverse evidence in log space is:")
        hm.logs.info_log(
            "ln_inv_evidence = {} +/- {}".format(
                ev.ln_evidence_inv, err_ln_inv_evidence
            )
        )
        hm.logs.info_log(
            "ln evidence = {} +/- {} {}".format(
                -ev.ln_evidence_inv, -err_ln_inv_evidence[1], -err_ln_inv_evidence[0]
            )
        )
        hm.logs.info_log("Analytic ln evidence is ", ln_evidence_analytic)
        delta = -ln_evidence_analytic - ev.ln_evidence_inv
        hm.logs.info_log(
            "Difference between analytic and harmonic  is ",
            delta,
            "+/-",
            err_ln_inv_evidence[0],
            err_ln_inv_evidence[1],
        )

        hm.logs.debug_log("kurtosis = {}".format(ev.kurtosis), " Aim for ~3.")
        check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
        hm.logs.debug_log("sqrt( var(var) ) / var = = {}".format(check))
        hm.logs.debug_log(
            " Aim for sqrt( 2/(n_eff-1) ) = {}".format(np.sqrt(2.0 / (ev.n_eff - 1)))
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

        # ======================================================================
        # Create corner/triangle plot.
        # ======================================================================
        if plot_corner and i_realisation == 0:
            hm.utils.plot_getdist(samples.reshape((-1, ndim)))
            if savefigs:
                save_name = save_name_start + "_gaussian_nondiagcov_getdist.png"
                plt.savefig(save_name, bbox_inches="tight")

            num_samp = chains_train.samples.shape[0]
            samps_compressed = model.sample(num_samp)

            hm.utils.plot_getdist_compare(chains_train.samples, samps_compressed)
            plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

            if savefigs:
                save_name = (
                    save_name_start
                    + "_gaussian_nondiagcov_corner_all_{}D.png".format(ndim)
                )
                plt.savefig(save_name, bbox_inches="tight", dpi=300)

            hm.utils.plot_getdist(samps_compressed)
            if savefigs:
                save_name = (
                    save_name_start
                    + "_gaussian_nondiagcov_flow_getdist_{}D.png".format(ndim)
                )
                plt.savefig(
                    save_name,
                    bbox_inches="tight",
                    dpi=300,
                )

            plt.show()

        # Save out realisations for violin plot.
        evidence_inv_summary[i_realisation, 0] = ev.evidence_inv
        evidence_inv_summary[i_realisation, 1] = ev.evidence_inv_var
        evidence_inv_summary[i_realisation, 2] = ev.evidence_inv_var_var

    clock = time.process_time() - clock
    hm.logs.info_log("Execution_time = {}s".format(clock))

    if n_realisations > 1:
        save_name = (
            save_name_start
            + "_gaussian_nondiagcov_evidence_inv"
            + "_realisations_{}D.dat".format(ndim)
        )
        np.savetxt(save_name, evidence_inv_summary)
        evidence_inv_analytic_summary = np.zeros(1)
        evidence_inv_analytic_summary[0] = np.exp(-ln_evidence_analytic)
        save_name = (
            save_name_start
            + "_gaussian_nondiagcov_evidence_inv"
            + "_analytic_{}D.dat".format(ndim)
        )
        np.savetxt(save_name, evidence_inv_analytic_summary)

    created_plots = True
    if created_plots:
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Setup logging config.
    hm.logs.setup_logging()

    # Define parameters.
    ndim = 21
    nchains = 100
    samples_per_chain = 5000
    # flow_str = "RealNVP"
    flow_str = "RQSpline"
    np.random.seed(10)  # used for initializing covariance matrix

    hm.logs.info_log("Non-diagonal Covariance Gaussian example")

    hm.logs.debug_log("-- Selected Parameters --")

    hm.logs.debug_log("Dimensionality = {}".format(ndim))
    hm.logs.debug_log("Number of chains = {}".format(nchains))
    hm.logs.debug_log("Samples per chain = {}".format(samples_per_chain))

    hm.logs.debug_log("-------------------------")

    # Run example.
    run_example(flow_str, ndim, nchains, samples_per_chain, plot_corner=False)
