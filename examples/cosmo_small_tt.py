import numpy as np
import time
import matplotlib.pyplot as plt
import harmonic as hm
import torch
import deep_tensor as dt
from functools import partial
import emcee
from typing import Callable


# JAX for the "small" cosmological simulator
import jax 
import jax.numpy as jnp
import jax_cosmo as jc

# Check available devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Torch using device:", device)
print("JAX devices:", jax.devices())

# Set precision
torch.set_default_dtype(torch.float64)
jax.config.update("jax_enable_x64", True)


def setup_jax_cosmo():
    # 1. Define Redshift bins
    nz1 = jc.redshift.smail_nz(1., 2.,  0.5)
    nz2 = jc.redshift.systematic_shift(jc.redshift.smail_nz(1., 2., 0.6), 0.5)
    nz3 = jc.redshift.systematic_shift(jc.redshift.smail_nz(1., 2., 0.8), 1.2)
    nz4 = jc.redshift.systematic_shift(jc.redshift.smail_nz(1., 2., 1.0), 1.8)
    nz5 = jc.redshift.systematic_shift(jc.redshift.smail_nz(1., 2., 1.2), 2.4)
    nzs = [nz1, nz2, nz3, nz4, nz5]

    # 2. Define probes and ell range
    probes = [jc.probes.WeakLensing(nzs, sigma_e=0.26)]
    ell = jnp.logspace(jnp.log10(2), jnp.log10(1000), 100)

    # 3. Create mock data
    cosmo_truth = jc.Planck15()
    cls_true = jc.angular_cl.angular_cl(cosmo_truth, ell, probes)
    noise_cls = jc.angular_cl.noise_cl(ell, probes)
    cov = jc.angular_cl.gaussian_cl_covariance(ell, probes, cls_true, noise_cls, sparse=False)
    
    # Generate one realization of noisy data to use
    mu = jnp.concatenate(cls_true)
    data_vector = jax.random.multivariate_normal(jax.random.PRNGKey(42), mu, cov)
    inv_cov = jnp.linalg.inv(cov)

    return ell, probes, data_vector, inv_cov, noise_cls

ELL, PROBES, DATA_OBS, INV_COV, NOISE_CLS = setup_jax_cosmo()

@jax.jit
def jax_likelihood(theta):
    """Log-posterior for the 5-parameter model. I guess this is the likelihood."""
    # Mapping theta to cosmology
    cosmo = jc.Cosmology(
        sigma8=theta[0], Omega_c=theta[1], Omega_b=theta[2],
        h=theta[3], n_s=theta[4],
        Omega_k=0.0, w0=-1.0, wa=0.0,
    )

    try:
        cls = jc.angular_cl.angular_cl(cosmo, ELL, PROBES)
        mu = jnp.concatenate(cls)
        diff = DATA_OBS - mu
        # Likelihood: -0.5 * chi^2
        lnlike = -0.5 * diff @ INV_COV @ diff
        return lnlike
    except:
        return -1e10 # Return very low value for unphysical regions

def ln_prior_box(theta, lower, upper):
    """Uniform Box Prior."""
    if np.any(theta < lower) or np.any(theta > upper):
        return -np.inf
    # Return log of the volume for a normalized prior
    return -np.log(np.prod(upper - lower))

def ln_posterior(theta, lower, upper):
    lp = ln_prior_box(theta, lower, upper)
    if not np.isfinite(lp):
        return -np.inf
    return lp + jax_likelihood(theta)


def run_small_cosmo_tt(
    nchains=50,
    samples_per_chain=1500,
    nburn=500,
    plot_corner=True,
):
    # Parameters from notebook: sigma8, Omega_c, Omega_b, h, n_s
    labels = ['sigma8', 'Omega_c', 'Omega_b', 'h', 'n_s']
    ndim = len(labels)
    
    limits = [
        (0.7, 0.9),   # sigma8 (Truth ~0.81)
        (0.1, 0.4),   # Omega_c (Truth ~0.26)
        (0.03, 0.06), # Omega_b (Truth ~0.049)
        (0.6, 0.9),   # h (Truth ~0.67)
        (0.7, 1.1),   # n_s (Truth ~0.96)
    ]

    # Extract lower and upper bounds from limits
    lower_tt, upper_tt = np.array(list(zip(*limits)))

    # Box bounds
    lower_prior = np.array([0.5, 0.1, 0.04, 0.64, 0.84])
    upper_prior = np.array([1.2, 0.5, 0.06, 0.82, 1.10])

    # Model parameters for RQSpline
    epochs_num = 40
    temperature = 0.9
    training_proportion = 0.5
    standardize = True
    n_layers = 4
    n_bins = 8
    hidden_size = [64, 64]
    # Spline range should roughly cover the prior domain
    spline_range = (-2.0, 2.0) 

    emcee_harmonic = True
    if emcee_harmonic:

        clock = time.process_time()

        # 1. Run Emcee
        print("Run sampling...")
        # Initialize walkers in a small ball around the center of the prior
        pos = lower_prior + (upper_prior - lower_prior) * (0.45 + 0.1 * np.random.rand(nchains, ndim))
        
        sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=[lower_prior, upper_prior])
        rstate = np.random.get_state()
        sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate, progress=True)
        
        samples_emcee = np.ascontiguousarray(sampler.chain[:, nburn:, :])
        lnprob_emcee = np.ascontiguousarray(sampler.lnprobability[:, nburn:])

        # Plot results
        hm.utils.plot_getdist(samples_emcee.reshape(-1, ndim), labels=labels)
        plt.savefig("small_cosmo_emcee_corner.png", bbox_inches="tight", dpi=300)
        plt.show()

        # 2. Configure chains for harmonic
        hm.logs.info_log("Configure chains...")
        chains = hm.Chains(ndim)
        chains.add_chains_3d(samples_emcee, lnprob_emcee)
        chains_train, chains_test = hm.utils.split_data(
            chains, training_proportion=training_proportion
        )

        # 3. Fit model
        print("Fit model for {} epochs...".format(epochs_num))
        model = hm.model.RQSplineModel(
            ndim,
            n_layers=n_layers,
            n_bins=n_bins,
            hidden_size=hidden_size,
            spline_range=spline_range,
            standardize=standardize,
            temperature=temperature,
        )
        model.fit(chains_train.samples, epochs=epochs_num, verbose=True)

        # 4. Computing evidence using learnt model
        print("Compute evidence...")
        ev = hm.Evidence(chains_test.nchains, model)
        ev.add_chains(chains_test)
        ln_evidence_hm =  -ev.ln_evidence_inv
        err_ln_inv_evidence_hm = ev.compute_ln_inv_evidence_errors()

        print("---------------------------------")
        print("Technical Details")
        print("---------------------------------")
        print(f"lnargmax = {ev.lnargmax}, lnargmin = {ev.lnargmin}")
        print(f"lnprobmax = {ev.lnprobmax}, lnprobmin = {ev.lnprobmin}")
        print(f"lnpredictmax = {ev.lnpredictmax}, lnpredictmin = {ev.lnpredictmin}")
        print("---------------------------------")
        print(f"shift = {ev.shift_value}, shift setting = {ev.shift}")
        
        print(f"ln_inv_evidence (harmonic)= {ev.ln_evidence_inv} +/- {err_ln_inv_evidence_hm}")
        print(f"ln evidence = {-ev.ln_evidence_inv} +/- {-err_ln_inv_evidence_hm[1]} {-err_ln_inv_evidence_hm[0]}")
        print(f"kurtosis = {ev.kurtosis} (Aim for ~3)")
        
        check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
        n_eff_limit = np.sqrt(2.0 / (ev.n_eff - 1))
        print(f"Standardized Variance Check: {check}")
        print(f"Aim for sqrt( 2/(n_eff-1) ) = {n_eff_limit}")
        print(f"sqrt(evidence_inv_var_var) / evidence_inv_var = {check}")

        # Visualise
        if plot_corner:
            # Model vs Data comparison
            num_samp = chains_train.samples.shape[0]
            samps_compressed = np.array(model.sample(num_samp))
            hm.utils.plot_getdist_compare(
                chains_train.samples, samps_compressed, labels=labels, legend_fontsize=12
            )
            plt.savefig("small_cosmo_hm_corner.png", bbox_inches="tight", dpi=300)
            plt.show()

        clock = time.process_time() - clock
        hm.logs.info_log("Execution time = {}s".format(clock))
    
    tt_evidence = True
    if tt_evidence:
        theta_ref = 0.5 * (lower_tt + upper_tt)
        lnpost_ref = float(ln_posterior(theta_ref, lower_tt, upper_tt))

        def neglog_posterior_torch_exact(theta_torch, lower, upper):
            theta_np = theta_torch.detach().cpu().numpy()
            lnps = np.array(
                [float(ln_posterior(t, lower, upper)) for t in theta_np],
                dtype=np.float64,
            )

            # True target for correction/evidence: no clipping, no arbitrary shift.
            lnps = np.where(np.isfinite(lnps), lnps, -1e30)
            neglogps = -lnps
            return torch.tensor(neglogps, dtype=theta_torch.dtype, device=theta_torch.device)

        def neglog_posterior_torch_tt(theta_torch, lower, upper):
            theta_np = theta_torch.detach().cpu().numpy()
            lnps = np.array(
                [float(ln_posterior(t, lower, upper)) for t in theta_np],
                dtype=np.float64,
            )

            # Deterministic penalty for invalid values
            lnps = np.where(np.isfinite(lnps), lnps, -1e30)

            # DIRT expects a negative log target. Shift by a fixed reference point
            # and clamp deterministically to keep TT fitting numerically stable.
            neglogps = np.clip(lnpost_ref - lnps, a_min=0.0, a_max=80.0)
            return torch.tensor(neglogps, dtype=theta_torch.dtype, device=theta_torch.device)

        # ===========================================================================
        # Configure Tensor Train (deep_tensor)
        # ===========================================================================
        hm.logs.info_log("Building Tensor Train approximation...")
        approximation_domain = torch.tensor(limits, dtype=torch.float64)

        # Create a partial function with lower and upper bounds pre-specified
        neglog_posterior_torch_tt_partial = partial(neglog_posterior_torch_tt, lower=lower_tt, upper=upper_tt)
        neglog_posterior_torch_exact_partial = partial(neglog_posterior_torch_exact, lower=lower_tt, upper=upper_tt)
        target_func = dt.TargetFunc(neglog_posterior_torch_tt_partial)
        
        reference = dt.UniformReference() 
        preconditioner = dt.UniformMapping(approximation_domain, reference)
        
        # More robust TT setup for the 5D cosmology posterior.
        tt_options = dt.TTOptions(max_als=6, init_rank=6, tt_method="fixed_rank")
        basis = dt.Lagrange1(num_elems=19)
        bases = dt.ApproxBases(basis, ndim)

        tt = dt.TT(tt_options)
        ftt = dt.FTT(bases, tt)
        bridge = dt.SingleLayer()
        dirt = dt.DIRT(target_func, preconditioner, ftt, bridge)

        hm.logs.info_log("Generating independent samples from TT...")
        num_sampl = nchains * (samples_per_chain-nburn)
        rs = reference.random(n=num_sampl, d=ndim)
        
        startTime = time.time()
        xs, neglogfxs_sirt = dirt.eval_irt(rs)
        neglogfxs_exact = neglog_posterior_torch_exact_partial(xs)
        res = dt.run_independence_sampler(xs, neglogfxs_sirt, neglogfxs_exact)
        
        print(f'Time to generate {num_sampl} samples: {(time.time()-startTime):.2f}s')
        print(f"Acceptance rate: {res.acceptance_rate:.3f}")

        samples_np = res.xs.detach().cpu().numpy()
        
        # Reshape for harmonic (nchains, nsamples, ndim)
        samples = samples_np.reshape(nchains, samples_per_chain-nburn, ndim)
        
        # Re-evaluate true log-probs for harmonic evidence calculation
        hm.logs.info_log("Computing exact log-probabilities for Harmonic...")
        lnprob = np.zeros((nchains, samples_per_chain-nburn))
        for i in range(nchains):
            for j in range(samples_per_chain-nburn):
                lnprob[i, j] = float(ln_posterior(samples[i, j], lower_tt, upper_tt))

        # ===========================================================================
        # Evidence Calculation (harmonic)
        # ===========================================================================
        hm.logs.info_log("Fitting Harmonic Flow model...")
        chains = hm.Chains(ndim)
        chains.add_chains_3d(samples, lnprob)
        chains_train, chains_test = hm.utils.split_data(chains, training_proportion=0.5)

        model = hm.model.RQSplineModel(ndim, temperature=0.9)
        model.fit(chains_train.samples, epochs=15, verbose=True, batch_size=256)

        ev = hm.Evidence(chains_test.nchains, model)
        ev.add_chains(chains_test)
        ln_evidence_hm_tt = -ev.ln_evidence_inv
        err_ln_inv_evidence_hm_tt = ev.compute_ln_inv_evidence_errors()

        print(f"Harmonic + tt posterior samples ln_evidence: {ln_evidence_hm_tt} +/- {-err_ln_inv_evidence_hm_tt[1]} {-err_ln_inv_evidence_hm_tt[0]}")

        print("---------------------------------")
        print("Technical Details")
        print("---------------------------------")
        print(f"lnargmax = {ev.lnargmax}, lnargmin = {ev.lnargmin}")
        print(f"lnprobmax = {ev.lnprobmax}, lnprobmin = {ev.lnprobmin}")
        print(f"lnpredictmax = {ev.lnpredictmax}, lnpredictmin = {ev.lnpredictmin}")
        print("---------------------------------")
        print(f"shift = {ev.shift_value}, shift setting = {ev.shift}")
        
        print(f"ln_inv_evidence (harmonic)= {ev.ln_evidence_inv} +/- {err_ln_inv_evidence_hm_tt}")
        print(f"ln evidence = {-ev.ln_evidence_inv} +/- {-err_ln_inv_evidence_hm_tt[1]} {-err_ln_inv_evidence_hm_tt[0]}")
        print(f"kurtosis = {ev.kurtosis} (Aim for ~3)")
        
        check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
        n_eff_limit = np.sqrt(2.0 / (ev.n_eff - 1))
        print(f"Standardized Variance Check: {check}")
        print(f"Aim for sqrt( 2/(n_eff-1) ) = {n_eff_limit}")
        print(f"sqrt(evidence_inv_var_var) / evidence_inv_var = {check}")
    

        if plot_corner:
            #Plot samples from tt
            hm.utils.plot_getdist(samples_np, labels=labels)
            plt.savefig("small_cosmo_tt_corner.png", bbox_inches="tight", dpi=300)
            plt.title("Samples from TT approximation")
            plt.show()

            #Plot trained flow vs TT samples
            flow_samples_tt = np.array(model.sample(samples_np.shape[0]))
            hm.utils.plot_getdist_compare(samples_np, flow_samples_tt, labels=labels)
            plt.savefig("small_cosmo_tt_vs_flow_corner.png", bbox_inches="tight", dpi=300)
            plt.show()
        
        def estimate_evidence(
            neglogpost: Callable[[torch.Tensor], torch.Tensor], 
            dirt: dt.DIRT, 
            num_samples: int
            ) -> torch.Tensor:
            """Computes an importance sampling estimate of the evidence."""
            
            # Generate a set of samples from DIRT approximation
            rs = dirt.reference.random(n=num_samples, d=ndim)
            xs, neglogposts_dirt = dirt.eval_irt(rs)
            
            # Evaluate the exact (unnormalised) posterior at each sample
            neglogposts_exact = neglogpost(xs)
            
            # Estimate evidence using importance sampling
            res = dt.run_importance_sampling(neglogposts_dirt, neglogposts_exact)
            evidence_estimate = res.log_norm.exp()
            
            return evidence_estimate

        clock = time.process_time()

        # Number of samples for importance sampling
        num_samples_tt = nchains * (samples_per_chain-nburn)

        # Estimate evidence
        evidence = estimate_evidence(neglog_posterior_torch_exact_partial, dirt, num_samples_tt)
        print(f"TT importance sampling evidence: {evidence.item():.4e}")
        clock = time.process_time() - clock
        print(f"TT importance sampling evidence estimation completed in {clock:.2f} seconds")

    if tt_evidence and emcee_harmonic:
        print("\nComparison of evidence estimates:")
        print(f"Harmonic + emcee estimate:{ln_evidence_hm:.4f} +/- {-err_ln_inv_evidence_hm[1]} {-err_ln_inv_evidence_hm[0]}")
        print(f"Harmonic + tt posterior samples ln_evidence: {ln_evidence_hm_tt} +/- {-err_ln_inv_evidence_hm_tt[1]} {-err_ln_inv_evidence_hm_tt[0]}")
        print(f" - TT importance sampling estimate: evidence = {evidence.item():.4e} (ln_evidence ~ {np.log(evidence.item()):.4f})")


if __name__ == "__main__":
    hm.logs.setup_logging()
    run_small_cosmo_tt(nchains=100, samples_per_chain=2000, nburn=500)