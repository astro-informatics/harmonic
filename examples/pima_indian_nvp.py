import numpy as np
import sys
import emcee
import scipy.special as sp
import time
import matplotlib.pyplot as plt
from functools import partial

sys.path.append(".")
import harmonic as hm

sys.path.append("examples")
import utils

sys.path.append("harmonic")
import model_nf
import flows


def ln_likelihood(y, theta, x):
    """Compute log_e of Pima Indian likelihood.

    Args:

        y: Vector of diabetes incidence (1=diabetes, 0=no diabetes).

        theta: Vector of parameter variables associated with covariates x.

        x: Vector of data covariates (e.g. NP, PGC, BP, TST, DMI etc.).

    Returns:

        double: Value of log_e likelihood at specified point in parameter
            space.

    """

    ln_p = compute_ln_p(theta, x)
    ln_pp = np.log(1.0 - np.exp(ln_p))
    return y.T.dot(ln_p) + (1 - y).T.dot(ln_pp)


def ln_prior(tau, theta):
    """Compute log_e of Pima Indian multivariate gaussian prior.

    Args:

        tau: Characteristic width of posterior \in [0.01,1].

        theta: Vector of parameter variables associated with covariates x.

    Returns:

        double: Value of log_e prior at specified point in parameter space.

    """

    d = len(theta)
    return 0.5 * d * np.log(tau / (2.0 * np.pi)) - 0.5 * tau * theta.T.dot(theta)


def ln_posterior(theta, tau, x, y):
    """Compute log_e of Pima Indian multivariate gaussian prior

    Args:

        theta: Vector of parameter variables associated with covariates x.

        tau: Characteristic width of posterior \in [0.01,1].

        x: Vector of data covariates (e.g. NP, PGC, BP, TST, DMI etc).

        y: Vector of incidence. 1=diabetes, 0=no diabetes.

    Returns:

        double: Value of log_e posterior at specified point in parameter
            space.

    """

    ln_pr = ln_prior(tau, theta)
    ln_L = ln_likelihood(y, theta, x)

    return ln_pr + ln_L


def compute_ln_p(theta, x):
    """Computes log_e probability ln(p) to be used in likelihood function.

    Args:

        theta: Vector of parameter variables associated with covariates x.

        x: Vector of data covariates (e.g. NP, PGC, BP, TST, DMI e.t.c.).

    Returns:

        double: Vector of the log-probabilities p to use in likelihood.

    """

    return -np.log(1.0 + 1.0 / np.exp(x.dot(theta)))


def run_example(
    model_1=True,
    tau=1.0,
    nchains=100,
    samples_per_chain=1000,
    nburn=500,
    plot_corner=False,
):
    """Run Pima Indians example.

    Args:

        model_1: Consider model 1 if true, otherwise model 2.

        tau: Precision parameter.

        nchains: Number of chains.

        samples_per_chain: Number of samples per chain.

        nburn: Number of burn in samples for each chain.

        plot_corner: Plot marginalised distributions if true.
    """

    # Set_dimension
    if model_1:
        ndim = 5
    else:
        ndim = 6

    hm.logs.debug_log("Dimensionality = {}".format(ndim))

    # ===========================================================================
    # Load Pima Indian data.
    # ===========================================================================
    hm.logs.info_log("Loading data ...")

    data = np.loadtxt("examples/data/pima_indian.dat")

    """
    Two primary models for comparison:
        Model 1: Uses rows const(1) + data(1,2,5,6) = 5 dimensional
        Model 2: Uses rows const(1) + data(1,2,5,6,7) = 6 dimensional
    data[:,0] --> Diabetes incidence. 
    data[:,1] --> Number of pregnancies (NP)
    data[:,2] --> Plasma glucose concentration (PGC)
    data[:,3] --> Diastolic blood pressure (BP)
    data[:,4] --> Tricept skin fold thickness (TST)
    data[:,5] --> Body mass index (BMI)
    data[:,6] --> Diabetes pedigree function (DP)
    data[:,7] --> Age (AGE)
    """
    x = np.zeros((len(data), ndim))

    if model_1:
        x[:, 0] = 1.0
        x[:, 1] = data[:, 1]
        x[:, 2] = data[:, 2]
        x[:, 3] = data[:, 5]
        x[:, 4] = data[:, 6]

    else:
        x[:, 0] = 1.0
        x[:, 1] = data[:, 1]
        x[:, 2] = data[:, 2]
        x[:, 3] = data[:, 5]
        x[:, 4] = data[:, 6]
        x[:, 5] = data[:, 7]  # --> model 2.

    """
    y[:] = 1 if patient has diabetes, 0 if patient does not have diabetes.
    """
    y = data[:, 0]

    """
    Configure some general parameters.
    """
    savefigs = True

    """
    Configure machine learning parameters.
    """

    training_proportion = 0.5
    var_scale = 0.9
    epochs_num = 50
    n_scaled = 6
    n_unscaled = 2

    # ===========================================================================
    # Compute random positions to draw from for emcee sampler.
    # ===========================================================================
    """
    Initial positions for each chain for each covariate \in [0,8).
    Simply drawn from directly from each covariate prior.
    """
    if model_1:
        pos_0 = np.random.randn(nchains) * 0.01
        pos_1 = np.random.randn(nchains) * 0.01
        pos_2 = np.random.randn(nchains) * 0.01
        pos_3 = np.random.randn(nchains) * 0.01
        pos_4 = np.random.randn(nchains) * 0.01
        pos = np.c_[pos_0, pos_1, pos_2, pos_3, pos_4]

    else:
        pos_0 = np.random.randn(nchains) * 0.01
        pos_1 = np.random.randn(nchains) * 0.01
        pos_2 = np.random.randn(nchains) * 0.01
        pos_3 = np.random.randn(nchains) * 0.01
        pos_4 = np.random.randn(nchains) * 0.01
        pos_5 = np.random.randn(nchains) * 0.01
        pos = np.c_[pos_0, pos_1, pos_2, pos_3, pos_4, pos_5]

    # Start Timer.
    clock = time.process_time()

    # ===========================================================================
    # Run Emcee to recover posterior samples
    # ===========================================================================
    hm.logs.info_log("Run sampling...")
    """
    Feed emcee the ln_posterior function, starting positions and recover chains.
    """
    sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=(tau, x, y))
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

    model = model_nf.RealNVPModel(
        ndim,
        n_scaled_layers=n_scaled,
        n_unscaled_layers=n_unscaled,
        temperature=var_scale,
    )
    model.fit(chains_train.samples, epochs=epochs_num)

    # =======================================================================
    # Visualise distributions
    # =======================================================================

    num_samp = chains_train.samples.shape[0]
    # samps = np.array(model.sample(num_samp, var_scale=1.))
    samps_compressed = np.array(model.sample(num_samp))

    labels = ["Bias", "NP", "PGC", "BMI", "DP", "AGE"]

    if model_1:
        model_lab = "model1"
        labels = labels[:-1]
    else:
        model_lab = "model2"

    utils.plot_getdist_compare(
        chains_train.samples, samps_compressed, labels=labels, legend_fontsize=17
    )

    if savefigs:
        plt.savefig(
            "examples/plots/nvp_pima_indian_corner_all_{}_T{}_tau{}_".format(
                n_scaled + n_unscaled, var_scale, tau
            )
            + model_lab
            + ".png",
            bbox_inches="tight",
            dpi=300,
        )

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
    hm.logs.info_log("Execution time = {}s".format(clock))

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


if __name__ == "__main__":
    # Setup logging config.
    hm.logs.setup_logging()

    # Define problem parameters
    model_1 = False
    # Tau should be varied in [0.01, 1].
    # tau = 1.0
    tau = 0.01

    # Define parameters.
    nchains = 200
    samples_per_chain = 5000
    nburn = 1000
    np.random.seed(3)

    hm.logs.info_log("Pima Indian example")

    if model_1:
        hm.logs.info_log("Using Model 1")
    else:
        hm.logs.info_log("Using Model 2")

    hm.logs.debug_log("-- Selected Parameters --")

    hm.logs.debug_log("Number of chains = {}".format(nchains))
    hm.logs.debug_log("Samples per chain = {}".format(samples_per_chain))
    hm.logs.debug_log("Burn in = {}".format(nburn))
    hm.logs.debug_log("Tau = {}".format(tau))

    hm.logs.debug_log("-------------------------")

    # Run example.
    samples = run_example(
        model_1, tau, nchains, samples_per_chain, nburn, plot_corner=True
    )
