import jax 
import torch
# Set precision
torch.set_default_dtype(torch.float64)
# Disable TensorFloat32 to ensure strict double precision
#torch.backends.cuda.matmul.allow_tf32 = False
#torch.backends.cudnn.allow_tf32 = False
jax.config.update("jax_enable_x64", True)
import numpy as np
import time
import matplotlib.pyplot as plt
import harmonic as hm
import torch
import deep_tensor as dt
# Check available devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Torch using device:", device)
import jax 
import jax.numpy as jnp
print("JAX devices:", jax.devices())
from functools import partial


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

        tau: Characteristic width of posterior \\in [0.01,1].

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

        tau: Characteristic width of posterior \\in [0.01,1].

        x: Vector of data covariates (e.g. NP, PGC, BP, TST, DMI etc).

        y: Vector of incidence. 1=diabetes, 0=no diabetes.

    Returns:

        double: Value of log_e posterior at specified point in parameter
            space.

    """

    ln_pr = ln_prior(tau, theta)
    ln_L = ln_likelihood(y, theta, x)

    return -(ln_pr + ln_L)


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
    
    ln_posterior_fixed = partial(ln_posterior, tau=tau, x=x, y=y)

    # Now ln_posterior_fixed only takes theta as argument
    # ln_posterior_fixed(theta) is equivalent to ln_posterior(theta, tau, x, y)

    # Wrapper for PyTorch Tensor Train
    def ln_posterior_torch(theta_torch):
        """
        theta_torch: torch.Tensor of shape (n_samples, ndim)
        Returns: torch.Tensor of shape (n_samples,)
        """
        # Convert to numpy
        theta_np = theta_torch.detach().cpu().numpy()
        
        # Evaluate for each sample
        logprobs = []
        for i in range(len(theta_np)):
            logprobs.append(float(ln_posterior_fixed(theta_np[i])))
        
        logprobs = np.array(logprobs)
        
        # Convert back to torch and negate
        return torch.tensor(logprobs, dtype=theta_torch.dtype, device=theta_torch.device)

    """
    Configure some general parameters.
    """
    savefigs = True

    """
    Configure machine learning parameters.
    """

    training_proportion = 0.5
    temperature = 0.9
    epochs_num = 60

    # "Bias", "NP", "PGC", "BMI", "DP", "AGE"
    # Model 1: 5 dimensions (Bias, NP, PGC, BMI, DP)
    # Conservative bounds
    approximation_domain_1 = torch.tensor([
        [-1.5, 0.5],    # Bias
        [-0.2, 1.2],    # NP
        [0.5, 2.0],     # PGC
        [-0.2, 1.0],    # BMI
        [0.0, 1.0]      # DP
    ], dtype=torch.float64)
    
    # Model 2: 6 dimensions (Bias, NP, PGC, BMI, DP, AGE)
    # Conservative bounds based on posterior plot (b)
    approximation_domain_2 = torch.tensor([
        [-1.5, 0.5],    # Bias
        [-0.2, 1.2],    # NP
        [0.5, 2.0],     # PGC
        [0.0, 1.2],     # BMI
        [0.0, 1.0],     # DP
        [-0.2, 0.8]     # AGE
    ], dtype=torch.float64)


    labels = ["Bias", "NP", "PGC", "BMI", "DP", "AGE"]

    if model_1:
        model_lab = "model1"
        labels = labels[:-1]
    else:
        model_lab = "model2"


    # Set up target function for TT
    target_func = dt.TargetFunc(ln_posterior_torch)
    # Start Timer.
    clock = time.process_time()

    reference = dt.UniformReference() # define reference measure
    # here you can choose different reference measure e.g. Gaussian
    if model_1:
        approximation_domain = approximation_domain_1
    else:
        approximation_domain= approximation_domain_2

    preconditioner = dt.UniformMapping(approximation_domain, reference) # define preconditioner
    tt_options = dt.TTOptions(max_als=1, init_rank=10, tt_method="fixed_rank") # set number of sweeps (max_als=1), ranks, fix ranks 

    basis = dt.Lagrange1(num_elems=50) # piecewise linear interpolation
    # here you can choose other interpolation basis such as fourier or chebyshev
    bases = dt.ApproxBases(basis, ndim) # set bases

    # may adjust earlier set options such as ranks number of sweep or allow increase of rank by not specifying tt_method="fixed_rank"
    # tt_options = dt.TTOptions(max_als=1, init_rank=10, tt_method="fixed_rank") # set number of sweeps (max_als=1), ranks, fix ranks
    tt = dt.TT(tt_options)
    ftt = dt.FTT(bases, tt)

    # defined above
    # target_func = dt.TargetFunc(multimodal_corr) # set target function
    bridge = dt.SingleLayer()  # set single-layer DIRT (i.e., SIRT)
    # do DIRT (layered) as  : dirt = dt.DIRT(target_func, preconditioner, ftt)#, bridge) # do single-layer DIRT (i.e., SIRT)
    sirt = dt.DIRT(target_func, preconditioner, ftt, bridge)

    # ===========================================================================
    # Generate samples
    # ===========================================================================
    hm.logs.info_log("Generate samples ...")

    startTime = time.time()
    num_sampl = nchains * samples_per_chain 
    # Draw a set of uniform random samples
    rs = reference.random(n=num_sampl, d=ndim)

    # Transform the samples according to SIRT approximation
    xs, neglogfxs_sirt = sirt.eval_irt(rs)
    # Compute potential function of the (unnormalised) target density at each SIRT sample
    neglogfxs_exact = target_func(xs)

    res = dt.run_independence_sampler(xs, neglogfxs_sirt, neglogfxs_exact)
    print(f'Time to gernerate {num_sampl} samples: {(time.time()-startTime):.2f}s')
    print("Samples shape:", res.xs.shape)


    print(f"Acceptance rate: {res.acceptance_rate:.3f}")
    print(f"IACT (x1): {res.iacts[0]:.3f}")
    print(f"IACT (x2): {res.iacts[1]:.3f}")
    print(f"IACT (x3): {res.iacts[2]:.3f}")
    print(f"IACT (x4): {res.iacts[3]:.3f}")
    if not model_1:
        print(f"IACT (x5): {res.iacts[4]:.3f}")


    samples_torch = res.xs  # PyTorch tensor
    samples_np = samples_torch.detach().numpy()  # Convert to numpy

    plot_samples = True
    if plot_samples:
        hm.utils.plot_getdist(
            samples_np, labels=labels
        )

        if savefigs:
            plt.savefig(
                "examples/plots/tt_pima_indian_samples_corner_tau{}_".format(tau
                )
                + model_lab
                + ".png",
                bbox_inches="tight",
                dpi=300,
            )
    
    samples = jnp.array(samples_np.reshape(nchains, samples_per_chain, ndim))
    
    print(f"Reshaped samples: {samples.shape}")
    print("Computing log probabilities...")
    
    if False:
        # Compute log probabilities - simple loop
        lnprob = np.zeros((nchains, samples_per_chain))
        for i in range(nchains):
            for j in range(samples_per_chain):
                lnprob[i, j] = -ln_posterior_fixed(samples[i, j, :])
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{nchains} chains...")


    lnprob = jnp.array(-res.potentials.detach().numpy().reshape(nchains, samples_per_chain))
    
    print(f"Log probabilities: {lnprob.shape}")
    

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

    model = hm.model.RQSplineModel(ndim, temperature=temperature)
    model.fit(chains_train.samples, epochs=epochs_num, verbose=True, batch_size=256)

    # =======================================================================
    # Visualise distributions
    # =======================================================================

    num_samp = chains_train.samples.shape[0]
    samps_compressed = np.array(model.sample(num_samp))

    if plot_corner:
        hm.utils.plot_getdist_compare(
            chains_train.samples, samps_compressed, labels=labels, legend_fontsize=17
        )

        if savefigs:
            plt.savefig(
                "examples/plots/tt_pima_indian_corner_all_T{}_tau{}_".format(
                 temperature, tau
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
    samples_per_chain = 3000
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
