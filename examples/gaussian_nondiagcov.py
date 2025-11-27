import numpy as np
import time
import matplotlib.pyplot as plt
import harmonic as hm
import jax
print(f"JAX is using these devices: {jax.devices()}")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import emcee
import logging


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
    thin=1,
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
        epochs_num = 10 #5
    elif flow_type == "RQSpline":
        #epochs_num = 5
        epochs_num = 110
    elif flow_type == "FlowMatching":
        # Longer training usually required; adjust if needed.
        epochs_num = 15000

    # Beginning of path where plots will be saved
    save_name_start = "examples/plots/" + flow_type

    temperature = 0.9
    standardize = False
    verbose = True
    
    # Spline params
    n_layers = 3
    n_bins = 128
    hidden_size = [32, 32]
    spline_range = (-10.0, 10.0)

    if flow_type == "RQSpline":
        save_name_start += "_"  + str(n_layers) + "l_" + str(n_bins) + "b_" + str(epochs_num) + "e_" + str(int(training_proportion * 100)) + "perc_" + str(temperature) + "T" + "_emcee"

    # Start timer.
    clock = time.process_time()

    # Run multiple realisations.
    n_realisations = 1
    ln_evidence_inv_summary = np.zeros((n_realisations, 5))
    for i_realisation in range(n_realisations):
        if n_realisations > 0:
            hm.logs.info_log(
                "Realisation = {}/{}".format(i_realisation + 1, n_realisations)
            )
        # Define the number of dimensions and the mean of the Gaussian
        num_samples = nchains * samples_per_chain
        # Initialize a PRNG key (you can use any valid key)
        key = jax.random.PRNGKey(i_realisation)
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
                pos, samples_per_chain, rstate0=rstate, progress=True
            )
            samples = np.ascontiguousarray(sampler.chain[:, nburn:, :])
            lnprob = np.ascontiguousarray(sampler.lnprobability[:, nburn:])
            
            # --- Thinning ---
            if thin > 1:
                samples = samples[:, ::thin, :]
                lnprob = lnprob[:, ::thin]


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

        elif flow_type == "FlowMatching":

            model = hm.model.FlowMatchingModel(
                ndim_in=ndim,
                hidden_dim=64,
                n_layers=5,
                learning_rate=1e-4,
                standardize=standardize,
                temperature=temperature,
            )
        model.fit(chains_train.samples, epochs=epochs_num, verbose=verbose, batch_size=2048)

        losses = np.array(model.loss_values)
        ema = None
        ema_beta = 0.99
        ema_losses = []
        for L in losses:
            ema = L if ema is None else ema_beta * ema + (1 - ema_beta) * L
            ema_losses.append(ema)
        plt.figure()
        plt.plot(losses, label="Loss")
        plt.plot(ema_losses, label="EMA", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        if savefigs:
            plt.savefig(save_name_start + "_gaussian_nondiagcov_loss.png", bbox_inches="tight", dpi=300)
        plt.show()
        plt.close()


        # Use chains and model to compute inverse evidence.
        hm.logs.info_log("Compute evidence...")

        # Process chains in batches to avoid memory issues
        batch_size = 10  # Number of chains to process at once
        n_test_chains = chains_test.nchains
        
        # Get samples and ln_posterior as 3D arrays
        # Reshape the flat arrays back to 3D
        samples_3d = []
        ln_posterior_3d = []
        for i_chain in range(n_test_chains):
            start, end = chains_test.get_chain_indices(i_chain)
            samples_3d.append(chains_test.samples[start:end, :])
            ln_posterior_3d.append(chains_test.ln_posterior[start:end])
        
        samples_3d = np.array(samples_3d)  # Shape: (n_test_chains, samples_per_chain, ndim)
        ln_posterior_3d = np.array(ln_posterior_3d)  # Shape: (n_test_chains, samples_per_chain)
        
        # Initialize Evidence object with first batch
        first_batch_size = min(batch_size, n_test_chains)
        first_batch_chains = hm.Chains(ndim)
        first_batch_chains.add_chains_3d(
            samples_3d[:first_batch_size],
            ln_posterior_3d[:first_batch_size]
        )
        
        ev = hm.Evidence(n_test_chains, model)
        ev.add_chains(first_batch_chains)
        hm.logs.info_log(f"Added batch 1/{(n_test_chains + batch_size - 1) // batch_size} ({first_batch_size} chains)")
        
        # Process remaining chains in batches
        for batch_start in range(batch_size, n_test_chains, batch_size):
            batch_end = min(batch_start + batch_size, n_test_chains)
            batch_chains = hm.Chains(ndim)
            batch_chains.add_chains_3d(
                samples_3d[batch_start:batch_end],
                ln_posterior_3d[batch_start:batch_end]
            )
            ev.add_chains(batch_chains)
            hm.logs.info_log(f"Added batch {(batch_start // batch_size) + 1}/{(n_test_chains + batch_size - 1) // batch_size} ({batch_end - batch_start} chains)")      
        
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
        hm.logs.info_log("Analytic ln evidence is {}".format(ln_evidence_analytic))
        delta = -ln_evidence_analytic - ev.ln_evidence_inv
        hm.logs.info_log(
            "Difference between analytic and harmonic  is {} +- {} {}".format(
                delta, err_ln_inv_evidence[0], err_ln_inv_evidence[1]
            )
        )

        hm.logs.info_log("kurtosis = {}".format(ev.kurtosis))
        hm.logs.info_log(" Aim for ~3.")
        check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
        hm.logs.info_log("sqrt( var(var) ) / var = {}".format(check))
        hm.logs.info_log(
            " Aim for sqrt( 2/(n_eff-1) ) = {}".format(np.sqrt(2.0 / (ev.n_eff - 1)))
        )

        # ===========================================================================
        # Display more technical details
        # ===========================================================================
        hm.logs.info_log("---------------------------------")
        hm.logs.info_log("Technical Details")
        hm.logs.info_log("---------------------------------")
        hm.logs.info_log(
            "lnargmax = {}, lnargmin = {}".format(ev.lnargmax, ev.lnargmin)
        )
        hm.logs.info_log(
            "lnprobmax = {}, lnprobmin = {}".format(ev.lnprobmax, ev.lnprobmin)
        )
        hm.logs.info_log(
            "lnpredictmax = {}, lnpredictmin = {}".format(
                ev.lnpredictmax, ev.lnpredictmin
            )
        )
        hm.logs.info_log("---------------------------------")
        hm.logs.info_log(
            "shift = {}, shift setting = {}".format(ev.shift_value, ev.shift)
        )
        hm.logs.info_log("running sum total = {}".format(sum(ev.running_sum)))
        hm.logs.info_log("running sum = \n{}".format(ev.running_sum))
        hm.logs.info_log("nsamples per chain = \n{}".format(ev.nsamples_per_chain))
        hm.logs.info_log(
            "nsamples eff per chain = \n{}".format(ev.nsamples_eff_per_chain)
        )
        hm.logs.info_log("===============================")

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
        ln_evidence_inv_summary[i_realisation, 0] = ev.ln_evidence_inv
        ln_evidence_inv_summary[i_realisation, 1] = err_ln_inv_evidence[0]
        ln_evidence_inv_summary[i_realisation, 2] = err_ln_inv_evidence[1]
        ln_evidence_inv_summary[i_realisation, 3] = ev.ln_evidence_inv_var
        ln_evidence_inv_summary[i_realisation, 4] = ev.ln_evidence_inv_var_var

    clock = time.process_time() - clock
    hm.logs.info_log("Execution_time = {}s".format(clock))

    if n_realisations > 1:
        save_name = (
            save_name_start
            + "_gaussian_nondiagcov_ln_evidence_inv"
            + "_realisations_{}D.dat".format(ndim)
        )
        np.savetxt(save_name, ln_evidence_inv_summary)
        ln_evidence_inv_analytic_summary = np.zeros(1)
        ln_evidence_inv_analytic_summary[0] = -ln_evidence_analytic
        save_name = (
            save_name_start
            + "_gaussian_nondiagcov_ln_evidence_inv"
            + "_analytic_{}D.dat".format(ndim)
        )
        np.savetxt(save_name, ln_evidence_inv_analytic_summary)

    created_plots = True
    if created_plots:
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Setup logging config.
    hm.logs.setup_logging(default_level=logging.DEBUG)

    # Define parameters.
    ndim = 50
    nchains = 200
    samples_per_chain = 500
    #flow_str = "RealNVP"
    #flow_str = "RQSpline"
    flow_str = "FlowMatching"
    np.random.seed(10)  # used for initializing covariance matrix

    hm.logs.info_log("Non-diagonal Covariance Gaussian example")
    hm.logs.info_log("-------------------------")
    hm.logs.info_log("Flow model: {}".format(flow_str))

    hm.logs.debug_log("-- Selected Parameters --")

    hm.logs.debug_log("Dimensionality = {}".format(ndim))
    hm.logs.debug_log("Number of chains = {}".format(nchains))
    hm.logs.debug_log("Samples per chain = {}".format(samples_per_chain))


    hm.logs.debug_log("-------------------------")

    # Run example.
    run_example(flow_str, ndim, nchains, samples_per_chain, plot_corner=False, thin=1)
