import numpy as np
import emcee
import scipy.special as sp
import time
import matplotlib.pyplot as plt
import harmonic as hm


def ln_likelihood(y, x, n, alpha, beta, tau):
    """Compute log_e of Radiata Pine likelihood.

    Args:

        y: Compression strength along grain.

        x: Predictor (density or density adjusted for resin content).

        alpha: Model bias term.

        beta: Model linear term.

        tau: Prior precision factor.

    Returns:

        double: Value of log_e likelihood at specified point in parameter
            space.

    """

    ln_like = 0.5 * n * np.log(tau)
    ln_like -= 0.5 * n * np.log(2.0 * np.pi)

    s = np.sum((y - alpha - beta * x) ** 2)

    ln_like -= 0.5 * tau * s

    return ln_like


def ln_prior_alpha(alpha, tau, mu_0, r_0):
    """Compute log_e of alpha / beta prior (Normal prior).

    Args:

        alpha: Model term (bias or linear term).

        tau: Prior precision factor.

        mu_0: Prior mean.

        r_0: Prior precision constant factor.

    Returns:

        double: Value of log_e prior at specified point in parameter space.

    """
    ln_pr_alpha = 0.5 * np.log(tau)
    ln_pr_alpha += 0.5 * np.log(r_0)
    ln_pr_alpha -= 0.5 * np.log(2.0 * np.pi)
    ln_pr_alpha -= 0.5 * tau * r_0 * (alpha - mu_0) ** 2

    return ln_pr_alpha


def ln_prior_tau(tau, a_0, b_0):
    """Compute log_e of tau prior (Gamma prior).

    Args:

        tau: Prior precision factor.

        a_0: Gamma prior shape parameter.

        b_0: Gamma prior rate parameter.

    Returns:

        double: Value of log_e tau prior at specified point in parameter
            space.

    """

    if tau < 0:
        return -np.inf

    ln_pr_tau = a_0 * np.log(b_0)
    ln_pr_tau += (a_0 - 1.0) * np.log(tau)
    ln_pr_tau -= b_0 * tau
    ln_pr_tau -= sp.gammaln(a_0)

    return ln_pr_tau


def ln_prior_separated(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0):
    """Compute log_e of prior (combining individual prior functions).

    Args:

        alpha: Model bias term.

        beta: Model linear term.

        tau: Prior precision factor.

        mu_0: Prior means.

        r_0: Prior precision constant factor for bias term.

        s_0: Prior precision constant factor for linear term.

        a_0: Gamma prior shape parameter.

        b_0: Gamma prior rate parameter.

    Returns:

        double: Value of log_e prior at specified point in parameter space.

    """
    ln_pr = ln_prior_alpha(alpha, tau, mu_0[0, 0], r_0)
    ln_pr += ln_prior_alpha(beta, tau, mu_0[1, 0], s_0)
    ln_pr += ln_prior_tau(tau, a_0, b_0)

    return ln_pr


def ln_prior_combined(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0):
    """Compute log_e of combined prior (jointly computing total prior).

    Args:

        alpha: Model bias term.

        beta: Model linear term.

        tau: Prior precision factor.

        mu_0: Prior means.

        r_0: Prior precision constant factor for bias term.

        s_0: Prior precision constant factor for linear term.

        a_0: Gamma prior shape parameter.

        b_0: Gamma prior rate parameter.

    Returns:

        double: Value of log_e prior at specified point in parameter space.

    """
    if tau < 0:
        return -np.inf

    ln_pr = a_0 * np.log(b_0)
    ln_pr += a_0 * np.log(tau)
    ln_pr -= b_0 * tau
    ln_pr -= np.log(2.0 * np.pi)
    ln_pr -= sp.gammaln(a_0)
    ln_pr += 0.5 * np.log(r_0)
    ln_pr += 0.5 * np.log(s_0)
    ln_pr -= (
        0.5 * tau * (r_0 * (alpha - mu_0[0, 0]) ** 2 + s_0 * (beta - mu_0[1, 0]) ** 2)
    )

    return ln_pr


def ln_prior(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0):
    """Compute log_e of combined prior.

    Can be used to easily switch with prior function using (e.g.
    ln_prior_separated or ln_prior_combined). There should be (and is) not
    difference (both implemented just as an additional consistency check).

    Args:

        alpha: Model bias term.

        beta: Model linear term.

        tau: Prior precision factor.

        mu_0: Prior means.

        r_0: Prior precision constant factor for bias term.

        s_0: Prior precision constant factor for linear term.

        a_0: Gamma prior shape parameter.

        b_0: Gamma prior rate parameter.

    Returns:

        double: Value of log_e prior at specified point in parameter space.

    """

    return ln_prior_combined(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0)


def ln_posterior(theta, y, x, n, mu_0, r_0, s_0, a_0, b_0):
    """Compute log_e of posterior.

    Args:

        theta: Position (alpha, beta, tau) at which to evaluate posterior.

        y: Compression strength along grain.

        x: Predictor (density or density adjusted for resin content).

        n: Number of specimens.

        mu_0: Prior means.

        r_0: Prior precision constant factor for bias term.

        s_0: Prior precision constant factor for linear term.

        a_0: Gamma prior shape parameter.

        b_0: Gamma prior rate parameter.

    Returns:

        double: Value of log_e posterior at specified theta (alpha, beta,
            tau) point.

    """

    alpha, beta, tau = theta

    ln_pr = ln_prior(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0)

    if not np.isfinite(ln_pr):
        return -np.inf

    ln_L = ln_likelihood(y, x, n, alpha, beta, tau)

    return ln_L + ln_pr


def ln_evidence_analytic(x, y, n, mu_0, r_0, s_0, a_0, b_0):
    """Compute log_e of analytic evidence.

    Args:

        x: Predictor (density or density adjusted for resin content).

        y: Compression strength along grain.

        n: Number of specimens.

        mu_0: Prior means.

        r_0: Prior precision constant factor for bias term.

        s_0: Prior precision constant factor for linear term.

        a_0: Gamma prior shape parameter.

        b_0: Gamma prior rate parameter.

    Returns:

        double: Value of log_e of analytic evidence for model.

    """

    Q_0 = np.diag([r_0, s_0])
    X = np.c_[np.ones((n, 1)), x]
    M = X.T.dot(X) + Q_0
    nu_0 = np.linalg.inv(M).dot(X.T.dot(y) + Q_0.dot(mu_0))

    quad_terms = y.T.dot(y) + mu_0.T.dot(Q_0).dot(mu_0) - nu_0.T.dot(M).dot(nu_0)

    ln_evidence = -0.5 * n * np.log(np.pi)
    ln_evidence += a_0 * np.log(2.0 * b_0)
    ln_evidence += sp.gammaln(0.5 * n + a_0) - sp.gammaln(a_0)
    ln_evidence += 0.5 * np.log(np.linalg.det(Q_0)) - 0.5 * np.log(np.linalg.det(M))

    ln_evidence += -(0.5 * n + a_0) * np.log(quad_terms + 2.0 * b_0)

    return ln_evidence


def run_example(
    flow_type,
    model_1=True,
    nchains=100,
    samples_per_chain=1000,
    nburn=500,
    plot_corner=False,
):
    """Run Radiata Pine example.

    Args:

        flow_type: Which flow model to use, "RealNVP" or "RQSpline".

        model_1: Consider model 1 if true, otherwise model 2.

        nchains: Number of chains.

        samples_per_chain: Number of samples per chain.

        nburn: Number of burn in samples for each chain.

        plot_corner: Plot marginalised distributions if true.
    """

    ndim = 3
    hm.logs.debug_log("Dimensionality = {}".format(ndim))

    # Set general parameters.
    savefigs = True

    # Beginning of path where plots will be saved
    save_name_start = "examples/plots/" + flow_type

    training_proportion = 0.5
    temperature = 0.8
    learning_rate = 0.001
    standardize = True

    # RealNVP parameters
    n_scaled = 3
    n_unscaled = 3

    # Spline parameters
    n_layers = 5
    n_bins = 5
    hidden_size = [32, 32]
    spline_range = (-10.0, 10.0)

    if flow_type == "RealNVP":
        epochs_num = 50
    if flow_type == "RQSpline":
        epochs_num = 30

    # ===========================================================================
    # Set-up Priors
    # ===========================================================================
    # Define prior variables
    mu_0 = np.array([[3000.0], [185.0]])
    r_0 = 0.06
    s_0 = 6.0
    a_0 = 3.0
    b_0 = 2.0 * 300**2

    hm.logs.debug_log("r_0 = {}".format(r_0))
    hm.logs.debug_log("s_0 = {}".format(s_0))
    hm.logs.debug_log("a_0 = {}".format(a_0))
    hm.logs.debug_log("b_0 = {}".format(b_0))

    # ===========================================================================
    # Load Radiata Pine data.
    # ===========================================================================
    hm.logs.info_log("Loading data ...")

    # Imports data file
    data = np.loadtxt("examples/data/radiata_pine.dat")
    id = data[:, 0]
    y = data[:, 1]
    x = data[:, 2]
    z = data[:, 3]
    n = len(x)

    # Ensure column vectors
    y = y.reshape(n, 1)
    x = x.reshape(n, 1)
    z = z.reshape(n, 1)

    # Remove means from covariates.
    x = x - np.mean(x)
    z = z - np.mean(z)

    # Set up and run sampler.
    tau_prior_mean = a_0 / b_0
    tau_prior_std = np.sqrt(a_0) / b_0

    # ===========================================================================
    # Compute random positions to draw from for emcee sampler.
    # ===========================================================================
    """
    Initial positions for each chain for each covariate \in [0,8).
    Simply drawn from directly from each covariate prior.
    """
    pos_alpha = mu_0[0, 0] + 1.0 / np.sqrt(tau_prior_mean * r_0) * np.random.randn(
        nchains
    )
    pos_beta = mu_0[1, 0] + 1.0 / np.sqrt(tau_prior_mean * s_0) * np.random.randn(
        nchains
    )
    pos_tau = tau_prior_mean + tau_prior_std * (
        np.random.rand(nchains) - 0.5
    )  # avoid negative tau

    """
    Concatenate these positions into a single variable 'pos'.
    """
    pos = np.c_[pos_alpha, pos_beta, pos_tau]

    # Start timer.
    clock = time.process_time()

    # ===========================================================================
    # Run Emcee to recover posterior samples
    # ===========================================================================
    hm.logs.info_log("Run sampling...")
    """
    Feed emcee the ln_posterior function, starting positions and recover chains.
    """
    if model_1:
        args = (y, x, n, mu_0, r_0, s_0, a_0, b_0)
    else:
        args = (y, z, n, mu_0, r_0, s_0, a_0, b_0)
    sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=args)
    rstate = np.random.get_state()
    sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
    samples = np.ascontiguousarray(sampler.chain[:, nburn:, :])
    lnprob = np.ascontiguousarray(sampler.lnprobability[:, nburn:])

    # ===========================================================================
    # Configure emcee chains for harmonic
    # ===========================================================================
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
    """
    Fit model by selecing the configuration of hyper-parameters which 
    minimises the validation variances.
    """

    if flow_type == "RealNVP":
        model = hm.model.RealNVPModel(
            ndim,
            n_scaled_layers=n_scaled,
            n_unscaled_layers=n_unscaled,
            learning_rate=learning_rate,
            standardize=standardize,
            temperature=temperature,
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
    model.fit(chains_train.samples, epochs=epochs_num)

    # ===========================================================================
    # Computing evidence using learnt model and emcee chains
    # ===========================================================================
    hm.logs.info_log("Compute evidence...")
    """
    Instantiates the evidence class with a given model. Adds some chains and 
    computes the log-space evidence (marginal likelihood).
    """
    ev = hm.Evidence(chains_test.nchains, model)
    ev.add_chains(chains_test)
    ln_evidence, ln_evidence_std = ev.compute_ln_evidence()
    evidence_std_log_space = (
        np.log(np.exp(ln_evidence) + np.exp(ln_evidence_std)) - ln_evidence
    )

    # ===========================================================================
    # End Timer.
    clock = time.process_time() - clock
    hm.logs.info_log("execution_time = {}s".format(clock))

    # ===========================================================================
    # Display evidence results
    # ===========================================================================
    hm.logs.info_log(
        "ln_evidence = {} +/- {}".format(ln_evidence, evidence_std_log_space)
    )
    hm.logs.info_log("kurtosis = {}".format(ev.kurtosis))
    hm.logs.info_log("sqrt( 2/(n_eff-1) ) = {}".format(np.sqrt(2.0 / (ev.n_eff - 1))))
    check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
    hm.logs.info_log("sqrt(evidence_inv_var_var) / evidence_inv_var = {}".format(check))
    ln_evidence_analytic_model1 = ln_evidence_analytic(
        x, y, n, mu_0, r_0, s_0, a_0, b_0
    )
    hm.logs.info_log(
        "ln_evidence_analytic_model1 = {}".format(ln_evidence_analytic_model1[0][0])
    )
    ln_evidence_analytic_model2 = ln_evidence_analytic(
        z, y, n, mu_0, r_0, s_0, a_0, b_0
    )
    hm.logs.info_log(
        "ln_evidence_analytic_model2 = {}".format(ln_evidence_analytic_model2[0][0])
    )

    # ===========================================================================
    # Display more technical details
    # ===========================================================================
    hm.logs.debug_log("---------------------------------")
    hm.logs.debug_log("Technical Details")
    hm.logs.debug_log("---------------------------------")
    hm.logs.debug_log("lnargmax = {}, lnargmin = {}".format(ev.lnargmax, ev.lnargmin))
    hm.logs.debug_log(
        "lnprobmax = {}, lnprobmin = {}".format(ev.lnprobmax, ev.lnprobmin)
    )
    hm.logs.debug_log(
        "lnpredictmax = {}, lnpredictmin = {}".format(ev.lnpredictmax, ev.lnpredictmin)
    )
    hm.logs.debug_log("---------------------------------")
    hm.logs.debug_log("shift = {}, shift setting = {}".format(ev.shift_value, ev.shift))
    hm.logs.debug_log("running sum total = {}".format(sum(ev.running_sum)))
    hm.logs.debug_log("running sum = \n{}".format(ev.running_sum))
    hm.logs.debug_log("nsamples per chain = \n{}".format(ev.nsamples_per_chain))
    hm.logs.debug_log("nsamples eff per chain = \n{}".format(ev.nsamples_eff_per_chain))
    hm.logs.debug_log("===============================")

    # ===========================================================================
    # Plotting and prediction functions
    # ===========================================================================

    # Create corner/triangle plot.
    created_plots = False
    if plot_corner:
        hm.utils.plot_getdist(samples.reshape((-1, ndim)))
        if savefigs:
            save_name = save_name_start + "_radiatapine_getdist.png"
            plt.savefig(save_name, bbox_inches="tight")

        plt.show(block=False)

        # =======================================================================
        # Visualise distributions
        # =======================================================================

        num_samp = chains_train.samples.shape[0]
        # samps = np.array(model.sample(num_samp, temperature=1.))
        samps_compressed = np.array(model.sample(num_samp))

        hm.utils.plot_getdist_compare(chains_train.samples, samps_compressed)
        if savefigs:
            save_name = save_name_start + "_radiatapine_getdist.png"
            plt.savefig(save_name, bbox_inches="tight")

        hm.utils.plot_getdist(samps_compressed)
        if savefigs:
            save_name = save_name_start + "_radiatapine_flow_getdist.png"
            plt.savefig(save_name, bbox_inches="tight")

        created_plots = True

    if created_plots:
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Setup logging config.
    hm.logs.setup_logging()

    # Define parameters.
    model_1 = True
    nchains = 400
    # nchains = 10
    # samples_per_chain = 20000
    samples_per_chain = 5000
    nburn = 500
    flow_str = "RealNVP"
    # flow_str = "RQSpline"
    # nburn = 100
    np.random.seed(2)

    hm.logs.info_log("Radiata Pine example")

    if model_1:
        hm.logs.info_log("Using Model 1")
    else:
        hm.logs.info_log("Using Model 2")

    hm.logs.debug_log("-- Selected Parameters --")

    hm.logs.debug_log("Number of chains = {}".format(nchains))
    hm.logs.debug_log("Samples per chain = {}".format(samples_per_chain))
    hm.logs.debug_log("Burn in = {}".format(nburn))

    hm.logs.debug_log("-------------------------")

    # Run example.
    samples = run_example(
        flow_str, model_1, nchains, samples_per_chain, nburn, plot_corner=True
    )
