import numpy as np
import time
import matplotlib.pyplot as plt
import harmonic as hm
import jax
jax.config.update("jax_enable_x64", True)
print(f"JAX is using these devices: {jax.devices()}")
import jax.numpy as jnp
import emcee
import logging


def ln_analytic_evidence(ndim, covs, weights):
    """Analytic log-evidence for unnormalised Gaussian mixture."""
    Z = 0
    for w, cov in zip(weights, covs):
        Z += w * (2 * np.pi) ** (ndim / 2) * np.linalg.det(cov) ** 0.5
    return np.log(Z)

def ln_posterior(x, means, inv_covs, weights):
    """
    Compute the unnormalised log posterior density of a Gaussian mixture at a point.

    This function evaluates the log of a weighted sum of (unnormalised) Gaussian
    component densities at position x. The normalisation constants of the Gaussian
    components (i.e. terms involving determinant of covariance and 2π factors) are
    omitted on purpose, so the returned value is only proportional to the true
    log posterior.

    Parameters
    ----------
    x : array-like (d,)
        Point in d-dimensional space where the posterior is evaluated.
        Should be a JAX array (jnp.ndarray) for compatibility with JAX transformations.
    means : array-like (K, d)
        Mean vectors of the K Gaussian mixture components.
    inv_covs : array-like (K, d, d)
        Inverse covariance matrices (precision matrices) for each component.
        Each must be symmetric positive definite.
    weights : array-like (K,)
        Mixture weights (prior probabilities) for each component. They should be
        non-negative and typically sum to 1, though the function does not enforce
        normalisation.

    Returns
    -------
    log_p : scalar (jnp.ndarray)
        The unnormalised log posterior value at x, computed as:
            log(sum_k w_k * exp(-0.5 * (x - μ_k)^T Σ_k^{-1} (x - μ_k)))
    """
    # Unnormalised log posterior for a Gaussian mixture
    logps = []
    for i in range(len(weights)):
        dx = x - means[i]
        # No normalization terms!
        logp = -jnp.dot(dx, jnp.dot(inv_covs[i], dx)) / 2.0
        logps.append(jnp.log(weights[i]) + logp)
    return jax.scipy.special.logsumexp(jnp.stack(logps))

def sample_mixture(key, means, covs, n_samples_per, ndim):
    """Sample from each Gaussian in the mixture."""
    samples = []
    lnprobs = []
    for i, (mean, cov) in enumerate(zip(means, covs)):
        subkey = jax.random.split(key, len(means))[i]
        s = jax.random.multivariate_normal(subkey, mean, cov, shape=(n_samples_per,))
        # Compute mixture logprob for each sample
        inv_covs = [jnp.linalg.inv(c) for c in covs]
        weights = jnp.ones(len(means)) / len(means)
        lps = jax.vmap(lambda x: ln_posterior(x, means, inv_covs, weights))(s)
        samples.append(s)
        lnprobs.append(lps)
    samples = jnp.concatenate(samples, axis=0)
    lnprobs = jnp.concatenate(lnprobs, axis=0)
    return samples, lnprobs


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
    burnin=500,
    plot_corner=False,
    thin=1,
    use_emcee=False,
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
    n_components = 5
    # Spread means in a cube instead of a line
    np.random.seed(42)  # For reproducible mean placement
    means = []
    for i in range(n_components):
        # Random position in [-2, 2]^ndim cube
        mean = (np.random.rand(ndim) - 0.5) * 4.0
        means.append(jnp.array(mean))
        
    covs = [jnp.array(init_cov(ndim))*0.01 for _ in range(n_components)]
    weights = jnp.ones(n_components) / n_components
    inv_covs = [jnp.linalg.inv(c) for c in covs]

    training_proportion = 0.5
    if flow_type == "RealNVP":
        epochs_num = 10 #5
    elif flow_type == "RQSpline":
        #epochs_num = 5
        epochs_num = 200
    elif flow_type == "FlowMatching":
        # Longer training usually required; adjust if needed.
        epochs_num = 8000
        # FlowMatching params
        hidden_dim = 256
        fm_n_layers = 8
        lr = 1e-4

    # Beginning of path where plots will be saved
    save_name_start = "examples/plots/" + flow_type 

    temperature = 0.95
    standardize = True
    verbose = True
    
    # Spline params
    n_layers = 8
    n_bins = 64
    hidden_size = [32, 32]
    spline_range = (-10.0, 10.0)

    if flow_type == "RQSpline":
        save_name_start += "_"  + str(n_layers) + "l_" + str(n_bins) + "b_" + str(epochs_num) + "e_" + str(int(training_proportion * 100)) + "perc_" + str(temperature) + "T" + "_emcee"
    if flow_type == "FlowMatching":
            save_name_start += "_hd" + str(hidden_dim) + "_nl" + str(fm_n_layers) + str(temperature) + "T" + "_{}D".format(ndim)

    # Start timer.
    clock = time.process_time()

    # Run multiple realisations.
    n_realisations = 100
    ln_evidence_inv_summary = np.zeros((n_realisations, 5))
    for i_realisation in range(n_realisations):
        if n_realisations > 0:
            hm.logs.info_log(
                "Realisation = {}/{}".format(i_realisation + 1, n_realisations)
            )
        
        
        if use_emcee:
            # ===== EMCEE SAMPLING =====
            print("Using emcee for sampling...")
            #np.random.seed(i_realisation)
            # Define log probability function for emcee
            def log_prob_emcee(x):
                return float(ln_posterior(jnp.array(x), means, inv_covs, weights))

            # Initialize walkers - equal number near each component
            nwalkers = nchains
            walkers_per_component = nwalkers // n_components
            remainder = nwalkers % n_components

            p0 = []
            for comp_idx in range(n_components):
                n_walkers_this = walkers_per_component + (1 if comp_idx < remainder else 0)
                for _ in range(n_walkers_this):
                    p0.append(np.array(means[comp_idx]) + 0.1 * np.random.randn(ndim))

            p0 = np.array(p0)
            hm.logs.info_log(f"Initialized {len(p0)} walkers: {walkers_per_component} per component (+{remainder} extra)")

            # Set up emcee sampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_emcee)

            # Run burn-in
            hm.logs.info_log(f"Running emcee burn-in for {burnin} steps...")
            state = sampler.run_mcmc(p0, burnin, progress=True)
            sampler.reset()

            # Run production
            hm.logs.info_log(f"Running emcee production for {samples_per_chain} steps...")
            sampler.run_mcmc(state, samples_per_chain, progress=True)

            # Get samples and log probabilities
            samples = sampler.get_chain(flat=False)
            lnprob = sampler.get_log_prob(flat=False)

            # Transpose to match expected format
            samples = np.transpose(samples, (1, 0, 2))
            lnprob = np.transpose(lnprob, (1, 0))

            # Convert to JAX arrays
            samples = jnp.array(samples)
            lnprob = jnp.array(lnprob)

            # Check acceptance fraction
            hm.logs.info_log(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
            
        else:
            # ===== DIRECT SAMPLING =====
            print("Using direct sampling...")
            key = jax.random.PRNGKey(i_realisation)
            total_samples_needed = nchains * samples_per_chain
            num_samples_per = (total_samples_needed + n_components - 1) // n_components
            samples, lnprob = sample_mixture(key, means, covs, num_samples_per, ndim)

            # Truncate to exact number needed
            samples = samples[:total_samples_needed]
            lnprob = lnprob[:total_samples_needed]

            # SHUFFLE before reshaping so each chain has mixture samples
            key, shuffle_key = jax.random.split(key)
            perm = jax.random.permutation(shuffle_key, total_samples_needed)
            samples = samples[perm]
            lnprob = lnprob[perm]

            samples = jnp.reshape(samples, (nchains, samples_per_chain, ndim))
            lnprob = jnp.reshape(lnprob, (nchains, samples_per_chain))

        # --- Thinning (common for both methods) ---
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
                hidden_dim=hidden_dim,
                n_layers=fm_n_layers,
                learning_rate= lr,
                standardize=standardize,
                temperature=temperature,
            )
        model.fit(chains_train.samples, epochs=epochs_num, verbose=verbose, batch_size=4096)

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
        plt.title("FlowMatching Training Loss")
        plt.legend()
        if savefigs:
            plt.savefig(save_name_start + "_gmm_loss.png", bbox_inches="tight", dpi=300)
        plt.close()

        plt.show()

        # Use chains and model to compute inverse evidence.
        hm.logs.info_log("Compute evidence...")

        ev = hm.Evidence(chains_test.nchains, model)
        # ev.set_mean_shift(0.0)
        ev.add_chains(chains_test)
        err_ln_inv_evidence = ev.compute_ln_inv_evidence_errors()

        # Compute analytic evidence.
        if i_realisation == 0:
            ln_evidence_analytic = ln_analytic_evidence(ndim, covs, weights)

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
                save_name = save_name_start + "_gmm_getdist.png"
                plt.savefig(save_name, bbox_inches="tight")

            num_samp = chains_train.samples.shape[0]
            samps_compressed = model.sample(num_samp)

            hm.utils.plot_getdist_compare(chains_train.samples, samps_compressed)
            plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

            if savefigs:
                save_name = (
                    save_name_start
                    + "_gmm_corner_all_{}D.png".format(ndim)
                )
                plt.savefig(save_name, bbox_inches="tight", dpi=300)

            hm.utils.plot_getdist(samps_compressed)
            if savefigs:
                save_name = (
                    save_name_start
                    + "_gmm_flow_getdist_{}D.png".format(ndim)
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
            + "_gmm_ln_evidence_inv"
            + "_realisations_{}D.dat".format(ndim)
        )
        np.savetxt(save_name, ln_evidence_inv_summary)
        ln_evidence_inv_analytic_summary = np.zeros(1)
        ln_evidence_inv_analytic_summary[0] = -ln_evidence_analytic
        save_name = (
            save_name_start
            + "_gmm_ln_evidence_inv"
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
    ndim = 20
    nchains = 200
    samples_per_chain = 2000
    burnin = 1000
    #flow_str = "RealNVP"
    #flow_str = "RQSpline"
    flow_str = "FlowMatching"
    np.random.seed(10)  # used for initializing covariance matrix

    hm.logs.info_log("Non-diagonal Covariance Gaussian example")

    hm.logs.debug_log("-- Selected Parameters --")

    hm.logs.debug_log("Dimensionality = {}".format(ndim))
    hm.logs.debug_log("Number of chains = {}".format(nchains))
    hm.logs.debug_log("Samples per chain = {}".format(samples_per_chain))

    hm.logs.debug_log("-------------------------")

    # Run example.
    run_example(flow_str, ndim, nchains, samples_per_chain, burnin, plot_corner=True, thin=5, use_emcee=False,)
