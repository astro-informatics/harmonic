import numpy as np
import emcee
import time
import matplotlib.pyplot as plt
from functools import partial
import harmonic as hm
import harmonic.logs as lg
from scipy.optimize import rosen


def ln_prior_uniform(x, xmin=-10.0, xmax=10.0, ymin=-5.0, ymax=15.0):
    """Compute log_e uniform prior.

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


def filter_outside_unit(x, log_l):
    # if x.ndim == 2:
    #    log_l = np.where(np.any((x < 0) | (x > 1), axis=-1), -np.inf, log_l)
    if np.any(x < 0) or np.any(x > 1):
        log_l = -np.inf
    return log_l


def rosenbrock_likelihood(x):

    log_l = -rosen((x.T - 0.5) * 10)

    return filter_outside_unit(x, log_l)


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
    flow_type, ndim=2, nchains=100, samples_per_chain=1000, nburn=500, plot_corner=False, thin=1
):
    """Run Rosenbrock example.

    Args:

        flow_type: Which flow model to use, "RealNVP" or "RQSpline".

        ndim: Dimension.

        nchains: Number of chains.

        samples_per_chain: Number of samples per chain.

        nburn: Number of burn in samples for each chain.

        plot_corner: Plot marginalised distributions if true.

        thin: Thinning factor for chains.
    """

    # if ndim != 2:
    #    raise ValueError("Only ndim=2 is supported (ndim={} specified)".format(ndim))

    # ===========================================================================
    # Configure Parameters.
    # ===========================================================================
    """
    Configure machine learning parameters
    """
    savefigs = True
    a = 1.0
    b = 100.0

    # Beginning of path where plots will be saved
    save_name_start = "examples/plots/" + flow_type + "_" + str(ndim) + "D"

    if flow_type == "RealNVP":
        epochs_num = 8
    elif flow_type == "RQSpline":
        # epochs_num = 5
        epochs_num = 400
    elif flow_type == "FlowMatching":
        epochs_num = 5000

    temperature = 0.8
    training_proportion = 0.5
    standardize = True
    # Spline params
    n_layers = 3
    n_bins = 8
    hidden_size = [32, 32]
    spline_range = (-10.0, 10.0)

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
    ln_evidence_inv_summary = np.zeros((n_realisations, 5))
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
        if ndim == 2:
            sampler = emcee.EnsembleSampler(
                nchains, ndim, ln_posterior, args=[ln_prior, a, b]
            )
        else:
            sampler = emcee.EnsembleSampler(nchains, ndim, rosenbrock_likelihood)

        rstate = np.random.get_state()
        sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
        samples = np.ascontiguousarray(sampler.chain[:, nburn::thin, :])
        lnprob  = np.ascontiguousarray(sampler.lnprobability[:, nburn::thin])

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
                hidden_dim=128,
                n_layers=5,
                learning_rate=1e-4,
                standardize=standardize,
                temperature=1.,
            )
        model.fit(chains_train.samples, epochs=epochs_num, verbose=True, batch_size=4096)
        model.temperature = temperature

        losses = np.array(model.loss_values)
        ema_beta = 0.99  # Smoothing factor
        ema_losses = []
        ema = None

        for loss in losses:
            if ema is None:
                ema = loss
            else:
                ema = ema_beta * ema + (1 - ema_beta) * loss
            ema_losses.append(ema)

        plt.plot(losses, label="Training Loss")
        plt.plot(ema_losses, color="red", label="EMA Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Flow Matching Training Loss")
        plt.legend()
        plt.show()


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
        err_ln_inv_evidence = ev.compute_ln_inv_evidence_errors()

        # Compute analytic evidence.
        if ndim == 2:
            hm.logs.debug_log("Compute evidence by numerical integration...")
            ln_posterior_func = partial(ln_posterior, ln_prior=ln_prior, a=a, b=b)
            ln_posterior_grid, x_grid, y_grid = hm.utils.eval_func_on_grid(
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
            hm.logs.info_log(
                "Evidence: numerical = {}, estimate = {}".format(
                    evidence_numerical_integration, np.exp(ln_evidence)
                )
            )

            hm.logs.info_log(
                "Ln evidence numerical = {}".format(
                    np.log(evidence_numerical_integration)
                )
            )

            hm.logs.debug_log(
                "Inv Evidence: numerical = {}, estimate = {}".format(
                    1.0 / evidence_numerical_integration,
                    np.exp(ev.ln_evidence_inv),
                )
            )
            diff = np.log(np.abs(evidence_numerical_integration - np.exp(ln_evidence)))
            hm.logs.info_log(
                "Evidence: |numerical - estimate| / estimate = {}".format(
                    np.exp(diff - ln_evidence)
                )
            )

            hm.logs.info_log(
                "Inv Evidence: |numerical - estimate| / estimate = {}".format(
                    np.abs(
                        1.0 / evidence_numerical_integration
                        - np.exp(ev.ln_evidence_inv)
                    )
                    / np.exp(ev.ln_evidence_inv)
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

        print(
            "ln_inv_evidence = {} +/- {}".format(
                ev.ln_evidence_inv, err_ln_inv_evidence
            )
        )
        print(
            "ln evidence = {} +/- {} {}".format(
                -ev.ln_evidence_inv, -err_ln_inv_evidence[1], -err_ln_inv_evidence[0]
            )
        )
        print("kurtosis = {}".format(ev.kurtosis), " Aim for ~3.")
        # print("ln inverse evidence per chain ", ev.ln_evidence_inv_per_chain)
        # print(
        #    "Average ln inverse evidence per chain ",#
        #    np.mean(ev.ln_evidence_inv_per_chain),
        # )
        print(
            "lnargmax",
            ev.lnargmax,
            "lnargmin",
            ev.lnargmin,
            "lnprobmax",
            ev.lnprobmax,
            "lnprobmin",
            ev.lnprobmin,
            "lnpredictmax",
            ev.lnpredictmax,
            "lnpredictmin",
            ev.lnpredictmin,
        )
        check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
        print(
            check,
            " Aim for sqrt( 2/(n_eff-1) ) = {}".format(np.sqrt(2.0 / (ev.n_eff - 1))),
        )
        print("sqrt(evidence_inv_var_var) / evidence_inv_var = {}".format(check))

        # Create corner/triangle plot.
        created_plots = False
        if plot_corner and i_realisation == 0:
            hm.utils.plot_getdist(samples.reshape((-1, ndim)))
            if savefigs:
                save_name = save_name_start + "_rosenbrock_getdist.png"
                plt.savefig(save_name, bbox_inches="tight")

            plt.show(block=False)

            # =======================================================================
            # Visualise distributions
            # =======================================================================

            num_samp = chains_train.samples.shape[0]
            samps_compressed = np.array(model.sample(num_samp))

            hm.utils.plot_getdist_compare(
                chains_train.samples, samps_compressed, legend_fontsize=12.5
            )
            if savefigs:
                save_name = (
                    save_name_start
                    + "_rosenbrock_corner_all_T"
                    + str(temperature)
                    + ".png"
                )
                plt.savefig(
                    save_name,
                    bbox_inches="tight",
                    dpi=300,
                )
            plt.show(block=False)

            model.temperature = 1.0
            samps_uncompressed = np.array(model.sample(num_samp))

            hm.utils.plot_getdist_compare(
                chains_train.samples, samps_uncompressed, legend_fontsize=12.5
            )
            if savefigs:
                save_name = (
                    save_name_start
                    + "_rosenbrock_corner_all_T1.png"
                )
                plt.savefig(
                    save_name,
                    bbox_inches="tight",
                    dpi=300,
                )
            plt.show(block=False)
            model.temperature = temperature
            created_plots = True

        ln_evidence_inv_summary[i_realisation, 0] = ev.ln_evidence_inv
        ln_evidence_inv_summary[i_realisation, 1] = err_ln_inv_evidence[0]
        ln_evidence_inv_summary[i_realisation, 2] = err_ln_inv_evidence[1]
        ln_evidence_inv_summary[i_realisation, 3] = ev.ln_evidence_inv_var
        ln_evidence_inv_summary[i_realisation, 4] = ev.ln_evidence_inv_var_var

    # ===========================================================================
    # End Timer.
    clock = time.process_time() - clock
    hm.logs.info_log("Execution time = {}s".format(clock))

    # ===========================================================================
    # Save out realisations of statistics for analysis.
    if n_realisations > 1:
        save_name = (
            save_name_start
            + "_rosenbrock_evidence_log_inv_T"
            + str(temperature)
            + "_realisations.dat"
        )
        np.savetxt(
            save_name,
            ln_evidence_inv_summary,
        )

        if ndim == 2:
            evidence_inv_analytic_summary = np.zeros(1)
            evidence_inv_analytic_summary[0] = -np.log(evidence_numerical_integration)
            save_name = save_name_start + "_rosenbrock_evidence_log_inv" + "_analytic.dat"
            np.savetxt(
                save_name,
                evidence_inv_analytic_summary,
            )

    if created_plots:
        input("\nPress Enter to continue...")

    return samples


if __name__ == "__main__":
    # Setup logging config.
    lg.setup_logging()

    # Define parameters.
    ndim = 2
    nchains = 30
    samples_per_chain = 5000
    nburn = 1000

    # flow_str = "RealNVP"
    flow_str = "RQSpline"
    np.random.seed(2)

    hm.logs.info_log("Rosenbrock example")

    hm.logs.debug_log("-- Selected Parameters --")

    hm.logs.debug_log("Dimensionality = {}".format(ndim))
    hm.logs.debug_log("Number of chains = {}".format(nchains))
    hm.logs.debug_log("Samples per chain = {}".format(samples_per_chain))
    hm.logs.debug_log("Burn in = {}".format(nburn))

    hm.logs.debug_log("-------------------------")

    # Run example.
    samples = run_example(
        flow_str, ndim, nchains, samples_per_chain, nburn, plot_corner=True, thin=5
    )
