import numpy as np
import emcee
import time
import matplotlib.pyplot as plt
from functools import partial
import harmonic as hm
import ex_utils
import jax
jax.config.update("jax_enable_x64", True)
print(f"JAX is using these devices: {jax.devices()}")


def ln_prior_uniform(x, xmin=-6.0, xmax=6.0, ymin=-6.0, ymax=6.0):
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


def ln_likelihood(x):
    """Compute log_e of likelihood defined by Rastrigin function.

    Args:

        x: Position at which to evaluate likelihood.

    Returns:

        double: Value of Rastrigin at specified point.

    """

    ndim = x.size

    f = 10.0 * ndim

    for i_dim in range(ndim):
        f += x[i_dim] ** 2 - 10.0 * np.cos(2.0 * np.pi * x[i_dim])

    return -f


def ln_posterior(x, ln_prior):
    """Compute log_e of posterior.

    Args:

        x: Position at which to evaluate posterior.

        ln_prior: Prior function.

    Returns:

        double: Posterior at specified point.

    """

    ln_L = ln_likelihood(x)

    if not np.isfinite(ln_L):
        return -np.inf
    else:
        return ln_prior(x) + ln_L


def run_example(
    model_type="Legacy",
    ndim=2,
    nchains=100,
    samples_per_chain=1000,
    nburn=500,
    plot_corner=False,
    plot_surface=False,
    thin=1,
):
    """Run Rastrigin example.

    Args:

        model_type: Which model to use "FlowMatching" or "RQSpline"

        ndim: Dimension.

        nchains: Number of chains.

        samples_per_chain: Number of samples per chain.

        nburn: Number of burn in samples for each chain.

        plot_corner: Plot marginalised distributions if true.

        plot_surface: Plot surface and samples if true.

        thin: Thinning factor for chains.

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
    temperature = 0.95
    standardize = True
    save_name_start = "examples/plots/" + model_type + "_s" + str(int(standardize)) + "_rastrigin_"
    nfold = 2
    nhyper = 2
    step = -2
    domain = []
    hyper_parameters = [[10 ** (R)] for R in range(-nhyper + step, step)]
    hm.logs.info_log("Hyper-parameters = {}".format(hyper_parameters))
    """
    Set prior parameters.
    """
    use_uniform_prior = True
    if use_uniform_prior:
        xmin = -6.0
        xmax = 6.0
        ymin = -6.0
        ymax = 6.0
        hm.logs.info_log(
            "xmin, xmax, ymin, ymax = {}, {}, {}, {}".format(xmin, xmax, ymin, ymax)
        )
        ln_prior = partial(ln_prior_uniform, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    # Start timer.
    clock = time.process_time()

    # ===========================================================================
    # Begin multiple realisations of estimator
    # ===========================================================================
    """
    Set up and run multiple simulations
    """
    n_realisations = 10
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
        pos = np.random.rand(ndim * nchains).reshape((nchains, ndim)) * 0.5
        sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=[ln_prior])
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
        chains_train, chains_test = hm.utils.split_data(chains, training_proportion=0.5)

        # =======================================================================
        # Fit model
        # =======================================================================
        hm.logs.info_log("Fit model...")

        if model_type == "FlowMatching":
            model = hm.model.FlowMatchingModel(
                ndim_in=ndim,
                hidden_dim=256,
                n_layers=6,
                learning_rate=1e-4,
                standardize=standardize,
                temperature=temperature,
            )
            model.fit(chains_train.samples, epochs=15000, verbose=True, batch_size=4096)

        elif model_type == "RQSpline":
            # Match the rosenbrock example’s spline setup
            n_layers = 3
            n_bins = 8
            hidden_size = [32, 32]
            spline_range = (-6.0, 6.0)  # Rastrigin domain

            model = hm.model.RQSplineModel(
                ndim,
                n_layers=n_layers,
                n_bins=n_bins,
                hidden_size=hidden_size,
                spline_range=spline_range,
                standardize=standardize,
                temperature=temperature,
            )
            model.fit(chains_train.samples, epochs=200, verbose=True, batch_size=4096)
        
        else:
            raise ValueError("Unsupported model_type: {}".format(model_type))
        

        # Plot training loss
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
        plt.title("Training Loss")
        plt.legend()
        if savefigs:
            save_name = (save_name_start + "_T"
                + str(temperature) + "loss.png"
            )
            plt.savefig(
                save_name,
                bbox_inches="tight",
                dpi=300,
            )
        plt.show(block=False)
        
        num_samp = chains_train.samples.shape[0]
        samps_compressed = np.array(model.sample(num_samp))

        hm.utils.plot_getdist_compare(
            chains_train.samples, samps_compressed, legend_fontsize=12.5
        )
        if savefigs:
            save_name = (
                save_name_start + "corner_all_T"
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
                save_name_start+ "corner_all_T1.png"
            )
            plt.savefig(
                save_name,
                bbox_inches="tight",
                dpi=300,
            )
        plt.show(block=False)
        model.temperature = temperature
        

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
            hm.logs.info_log("Compute evidence by numerical integration...")
            ln_posterior_func = partial(ln_posterior, ln_prior=ln_prior)
            ln_posterior_grid, x_grid, y_grid = hm.utils.eval_func_on_grid(
                ln_posterior_func,
                xmin=-6.0,
                xmax=6.0,
                ymin=-6.0,
                ymax=6.0,
                nx=1000,
                ny=1000,
            )
            dx = x_grid[0, 1] - x_grid[0, 0]
            dy = y_grid[1, 0] - y_grid[0, 0]
            evidence_numerical_integration = np.sum(np.exp(ln_posterior_grid)) * dx * dy

        # ======================================================================
        # Display evidence computation results (log-space like Rosenbrock).
        # ======================================================================
        hm.logs.info_log(
            "Evidence: numerical = {}, estimate = {}".format(
                evidence_numerical_integration, np.exp(ln_evidence)
            )
        )
        hm.logs.info_log("Ln evidence numerical = {}".format(np.log(evidence_numerical_integration)))
        hm.logs.info_log("Ln evidence estimate = {}".format(ln_evidence))

        diff = np.log(np.abs(evidence_numerical_integration - np.exp(ln_evidence)))
        hm.logs.info_log(
            "Evidence: |numerical - estimate| / estimate = {}".format(
                np.exp(diff - ln_evidence)
            )
        )

        hm.logs.info_log(
            "Inv Evidence: numerical = {}, estimate = {}".format(
                1.0 / evidence_numerical_integration, np.exp(ev.ln_evidence_inv)
            )
        )
        hm.logs.info_log(
            "Inv Evidence: |numerical - estimate| / estimate = {}".format(
                np.abs(1.0 / evidence_numerical_integration - np.exp(ev.ln_evidence_inv))
                / np.exp(ev.ln_evidence_inv)
            )
        )

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
        print("kurtosis = {}".format(ev.kurtosis), " Aim for ~3.")
        check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
        print(
            check,
            " Aim for sqrt( 2/(n_eff-1) ) = {}".format(np.sqrt(2.0 / (ev.n_eff - 1))),
        )
        print("sqrt(evidence_inv_var_var) / evidence_inv_var = {}".format(check))
        hm.logs.info_log("===============================")

        # Create corner/triangle plot.
        created_plots = False
        if plot_corner and i_realisation == 0:
            ex_utils.plot_corner(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig(save_name_start + "corner.png", bbox_inches="tight")

            hm.utils.plot_getdist(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig(save_name_start + "getdist.png", bbox_inches="tight")

            plt.show(block=False)
            created_plots = True

        # In 2D case, plot surface/image and samples.
        if plot_surface and ndim == 2 and i_realisation == 0:
            # Plot ln_posterior surface.
            # ln_posterior_grid[ln_posterior_grid<-100.0] = -100.0
            i_chain = 0
            ax = ex_utils.plot_surface(
                ln_posterior_grid,
                x_grid,
                y_grid,
                samples[i_chain, :, :].reshape((-1, ndim)),
                lnprob[i_chain, :].reshape((-1, 1)),
            )
            # ax.set_zlim(-100.0, 0.0)
            ax.set_zlabel(r"$\log \mathcal{L}$")
            if savefigs:
                plt.savefig(
                    "examples/plots/rastrigin_lnposterior_surface.png",
                    bbox_inches="tight",
                )

            # Plot posterior image.
            ax = ex_utils.plot_image(
                np.exp(ln_posterior_grid),
                x_grid,
                y_grid,
                samples.reshape((-1, ndim)),
                colorbar_label=r"$\mathcal{L}$",
            )
            # ax.set_clim(vmin=0.0, vmax=0.003)
            if savefigs:
                plt.savefig(
                    "examples/plots/rastrigin_posterior_image.png", bbox_inches="tight"
                )

            # Evaluate model on grid.
            model_grid, x_grid, y_grid = hm.utils.eval_func_on_grid(
                model.predict,
                xmin=-6.0,
                xmax=6.0,
                ymin=-6.0,
                ymax=6.0,
                nx=1000,
                ny=1000,
            )
            # model_grid[model_grid<-100.0] = -100.0

            # Plot model.
            ax = ex_utils.plot_image(
                model_grid, x_grid, y_grid, colorbar_label=r"$\log \varphi$"
            )
            # ax.set_clim(vmin=-2.0, vmax=2.0)
            if savefigs:
                plt.savefig(
                    "examples/plots/rastrigin_model_image.png", bbox_inches="tight"
                )

            # Plot exponential of model.
            ax = ex_utils.plot_image(
                np.exp(model_grid), x_grid, y_grid, colorbar_label=r"$\varphi$"
            )
            # ax.set_clim(vmin=0.0, vmax=6.0)
            if savefigs:
                plt.savefig(
                    "examples/plots/rastrigin_modelexp_image.png", bbox_inches="tight"
                )

                plt.show(block=False)
                created_plots = True

        # Save out realisations for voilin plot (log-space, like Rosenbrock).
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
    # Save out realisations of statistics for analysis (log-space).
    if n_realisations > 1:
        save_name = (
            save_name_start
            + "rastrigin_evidence_log_inv_T"
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
            save_name = save_name_start + "rastrigin_evidence_log_inv_analytic.dat"
            np.savetxt(
                save_name,
                evidence_inv_analytic_summary,
            )

    if created_plots:
        input("\nPress Enter to continue...")

    return samples


if __name__ == "__main__":
    # Setup logging config.
    import logging
    logging.basicConfig(level=logging.INFO)
    hm.logs.setup_logging()    # Keep Harmonic debug messages
    logging.getLogger("Harmonic").setLevel(logging.DEBUG)


    # Optionally: generic suppression for any future jax.* logger
    for lname in list(logging.root.manager.loggerDict.keys()):
        if lname.startswith("jax."):
            logging.getLogger(lname).setLevel(logging.WARNING)

    # Define parameters.
    ndim = 2
    nchains = 80
    samples_per_chain = 5000
    nburn = 2000
    architecture ="FlowMatching"  # "RQSpline" or "FlowMatching"
    np.random.seed(20)

    hm.logs.info_log("Rastrigin example")

    hm.logs.info_log("-- Selected Parameters --")

    hm.logs.info_log("Dimensionality = {}".format(ndim))
    hm.logs.info_log("Number of chains = {}".format(nchains))
    hm.logs.info_log("Samples per chain = {}".format(samples_per_chain))
    hm.logs.info_log("Burn in = {}".format(nburn))
    hm.logs.info_log("Architecture = {}".format(architecture))

    hm.logs.info_log("-------------------------")

    # Run example.
    samples = run_example(architecture,
        ndim, nchains, samples_per_chain, nburn, plot_corner=True, plot_surface=False, thin=10
    )
