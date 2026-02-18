import numpy as np
import time
import matplotlib.pyplot as plt
import harmonic as hm
import torch
import deep_tensor as dt
import desy1
from functools import partial

# Check available devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Torch using device:", device)
import jax 
import jax.numpy as jnp
print("JAX devices:", jax.devices())

# Set double precision
torch.set_default_dtype(torch.float32)
jax.config.update("jax_enable_x64", False)


def run_cosmo_tt_example(
    nchains=100,
    samples_per_chain=1000,
    plot_corner=False,
):
    """Run cosmological tensor train example.

    Args:
        nchains: Number of chains.
        samples_per_chain: Number of samples per chain.
        plot_corner: Plot marginalised distributions if true.
    """

    # Cosmological parameters (21 dimensions)
    input_params = [
        "sigma8",
        "Omega_c", 
        "Omega_b",
        "h",
        "n_s",
        "m1", "m2", "m3", "m4",
        "dz1", "dz2", "dz3", "dz4", 
        "A",
        "eta",
        "bias1", "bias2", "bias3", "bias4", "bias5",
    ]
    
    ndim = len(input_params)
    hm.logs.debug_log("Dimensionality = {}".format(ndim))

    # Parameter limits
    limits_old = [
        (0.5, 0.9),     # sigma8
        (0.1, 0.5),     # Omega_c
        (0.03, 0.06),   # Omega_b
        (0.5, 0.9),     # h
        (0.9, 1.05),    # n_s
        (-0.06, 0.06),  # m1
        (-0.06, 0.06),  # m2
        (-0.06, 0.06),  # m3
        (-0.06, 0.06),  # m4
        (-0.1, 0.1),    # dz1
        (-0.1, 0.1),    # dz2
        (-0.1, 0.1),    # dz3
        (-0.1, 0.1),    # dz4
        (0.0, 3.0),     # A
        (-3., 3.),      # eta
        (0.8, 3.0),     # bias1
        (0.8, 3.0),     # bias2
        (0.8, 3.0),     # bias3
        (0.8, 3.0),     # bias4
        (0.8, 3.0),     # bias5
    ]

    limits = [
    (0.78, 0.82),     # sigma8
    (0.22, 0.28),     # Omega_c
    (0.03, 0.06),     # Omega_b
    (0.55, 0.85),     # h
    (0.93, 1.02),     # n_s
    (-0.06, 0.06),    # m1
    (-0.06, 0.06),    # m2
    (-0.06, 0.06),    # m3
    (-0.05, 0.07),    # m4
    (-0.03, 0.03),    # dz1
    (-0.03, 0.02),    # dz2
    (-0.02, 0.03),    # dz3
    (-0.04, 0.04),    # dz4
    (0.30, 0.70),     # A
    (-3.0, 3.0),      # eta
    (1.14, 1.26),     # bias1
    (1.33, 1.47),     # bias2
    (1.53, 1.67),     # bias3
    (1.73, 1.87),     # bias4
    (1.92, 2.08),     # bias5
    ]

    # ===========================================================================
    # Setup cosmological posterior
    # ===========================================================================
    hm.logs.info_log("Setting up cosmological posterior...")
    
    y1 = desy1.MockY1Likelihood()
    
    def ln_posterior_fixed(theta):
        return -y1.posterior(theta) 

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
        print("Max logprob (torch wrapper):", np.max(logprobs))
        print("Min logprob (torch wrapper):", np.min(logprobs))
        
        return torch.tensor(logprobs, dtype=theta_torch.dtype, device=theta_torch.device)


    def ln_posterior_torch_maybe(theta_torch):
        theta_np = theta_torch.detach().cpu().numpy()
        
        # 1. Get raw log-probs (e.g., -570, -600, -5000)
        raw_lp = np.array([float(ln_posterior_fixed(t)) for t in theta_np])
        
        # 2. THE SHIFT: Force the maximum value in this batch to be 0.0
        # This ensures e^0 = 1.0. No more underflow.
        current_max = np.max(raw_lp)
        shifted_lp = raw_lp - current_max
        
        # 3. THE CLAMP: 
        # A log-prob of -50 is already 10^-22 times less likely than the peak.
        # Going lower than -50 provides no useful info to the TT and only 
        # causes the SVD matrix to be "ill-conditioned."
        final_lp = np.clip(shifted_lp, a_min=-50.0, a_max=None)
        
        print(f"SVD Input -> Max: {np.max(final_lp):.1f}, Min: {np.min(final_lp):.1f} (Shifted by {-current_max:.1f})")
        
        return torch.tensor(final_lp, dtype=theta_torch.dtype, device=theta_torch.device)

    # ===========================================================================
    # Configure tensor train parameters
    # ===========================================================================
    
    # Convert limits to approximation domain
    approximation_domain = torch.tensor(limits, dtype=torch.float32)
    
    labels = input_params

    # Set up target function for TT
    target_func = dt.TargetFunc(ln_posterior_torch)
    
    # Start Timer
    clock = time.process_time()

    reference = dt.UniformReference()  # define reference measure
    preconditioner = dt.UniformMapping(approximation_domain, reference)  # define preconditioner
    
    tt_options = dt.TTOptions(max_als=5, init_rank=20, tt_method="fixed_rank")
    
    basis = dt.Lagrange1(num_elems=30)  # piecewise linear interpolation - reduced for 21D
    bases = dt.ApproxBases(basis, ndim)  # set bases

    tt = dt.TT(tt_options)
    ftt = dt.FTT(bases, tt)

    bridge = dt.SingleLayer()  # set single-layer DIRT (i.e., SIRT)
    dirt = dt.DIRT(target_func, preconditioner, ftt, bridge)


    if False:
        hm.logs.info_log("Computing evidence from TT cores...")
        reduced_cores = []
        evidence = torch.eye(1, dtype=torch.float32, device=device)  # Initialize evidence as 1
        for k in range(ndim):
            core_k = dirt.sirts[0].ftt.tt.cores[k]
            reduced_core_k = core_k.sum(dim=1)**2  # Second dimension sum
            evidence = evidence @ reduced_core_k.clone()
            reduced_cores.append(reduced_core_k)
            print(f"Core {k}: shape {core_k.shape}")

        print("Result shape:", evidence.shape)
        print(f"Evidence from TT cores integration: {evidence}")
        print(f"Log evidence from TT cores integration: {torch.log(evidence)}")

    # ===========================================================================
    # Generate samples
    # ===========================================================================
    hm.logs.info_log("Generate samples...")

    startTime = time.time()
    num_sampl = nchains * samples_per_chain 
    
    # Draw a set of uniform random samples
    rs = reference.random(n=num_sampl, d=ndim)

    # Transform the samples according to SIRT approximation
    xs, neglogfxs_sirt = dirt.eval_irt(rs)
    # Compute potential function of the (unnormalised) target density at each SIRT sample
    neglogfxs_exact = target_func(xs)

    res = dt.run_independence_sampler(xs, neglogfxs_sirt, neglogfxs_exact)
    print(f'Time to generate {num_sampl} samples: {(time.time()-startTime):.2f}s')
    print("Samples shape:", res.xs.shape)

    print(f"Acceptance rate: {res.acceptance_rate:.3f}")
    print(f"IACT (sigma8): {res.iacts[0]:.3f}")
    print(f"IACT (Omega_c): {res.iacts[1]:.3f}")
    print(f"IACT (Omega_b): {res.iacts[2]:.3f}")
    print(f"IACT (h): {res.iacts[3]:.3f}")
    print(f"IACT (n_s): {res.iacts[4]:.3f}")

    samples_torch = res.xs  # PyTorch tensor
    samples_np = samples_torch.detach().cpu().numpy()  # Convert to numpy

    # Plot samples
    plot_samples = True
    if plot_samples:
        hm.utils.plot_getdist(
            samples_np, labels=labels[:6]  # Plot first 6 parameters only for visibility
        )

        plt.savefig(
            "cosmo_tt_samples_corner.png",
            bbox_inches="tight",
            dpi=300,
        )
    
    # Reshape for harmonic
    samples = samples_np.reshape(nchains, samples_per_chain, ndim)
    
    print(f"Reshaped samples: {samples.shape}")
    print("Computing log probabilities...")
    
    # Compute log probabilities - simple loop
    lnprob = np.zeros((nchains, samples_per_chain))
    for i in range(nchains):
        for j in range(samples_per_chain):
            lnprob[i, j] = ln_posterior_fixed(samples[i, j, :])
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{nchains} chains...")
    
    print(f"Log probabilities: {lnprob.shape}")

    # ===========================================================================
    # Configure chains for harmonic
    # ===========================================================================
    hm.logs.info_log("Configure chains...")
    
    training_proportion = 0.5
    chains = hm.Chains(ndim)
    chains.add_chains_3d(samples, lnprob)
    chains_train, chains_test = hm.utils.split_data(
        chains, training_proportion=training_proportion
    )

    # ===========================================================================
    # Fit model
    # ===========================================================================
    temperature = 0.9
    epochs_num = 20
    
    hm.logs.info_log("Fit model for {} epochs...".format(epochs_num))
    
    model = hm.model.RQSplineModel(ndim, temperature=temperature)
    model.fit(chains_train.samples, epochs=epochs_num, verbose=True, batch_size=256)

    # ===========================================================================
    # Visualise distributions (optional)
    # ===========================================================================
    num_samp = chains_train.samples.shape[0]
    samps_compressed = np.array(model.sample(num_samp))

    if plot_corner:
        hm.utils.plot_getdist_compare(
            chains_train.samples, samps_compressed, labels=labels[:6], legend_fontsize=17
        )

        plt.savefig(
            "cosmo_tt_corner_comparison.png",
            bbox_inches="tight",
            dpi=300,
        )

    # ===========================================================================
    # Computing evidence using learnt model
    # ===========================================================================
    hm.logs.info_log("Compute evidence...")
    
    ev = hm.Evidence(chains_test.nchains, model)
    ev.add_chains(chains_test)
    ln_evidence, ln_evidence_std = ev.compute_ln_evidence()
    evidence_std_log_space = (
        np.log(np.exp(ln_evidence) + np.exp(ln_evidence_std)) - ln_evidence
    )

    # ===========================================================================
    # End Timer
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

    return samples_np


if __name__ == "__main__":
    # Setup logging config
    hm.logs.setup_logging()

    # Define parameters
    nchains = 50  # Reduced for 21D problem
    samples_per_chain = 2000
    np.random.seed(42)

    hm.logs.info_log("Cosmological Tensor Train example")
    
    hm.logs.debug_log("-- Selected Parameters --")
    hm.logs.debug_log("Number of chains = {}".format(nchains))
    hm.logs.debug_log("Samples per chain = {}".format(samples_per_chain))
    hm.logs.debug_log("-------------------------")

    # Run example
    samples = run_cosmo_tt_example(
        nchains, samples_per_chain, plot_corner=True
    )