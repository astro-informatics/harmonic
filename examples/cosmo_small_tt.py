import torch
import jax 
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
import deep_tensor as dt
from functools import partial
import emcee
from typing import Callable
import jax.numpy as jnp
import jax_cosmo as jc
import os

# Check available devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Torch using device:", device)
print("JAX devices:", jax.devices())

# Invalid likelihood penalty for unphysical regions
INVALID_LOGLIKE_RAW = -1e30

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

    cls = jc.angular_cl.angular_cl(cosmo, ELL, PROBES)
    mu = jnp.concatenate(cls)
    diff = DATA_OBS - mu
    # Likelihood: -0.5 * chi^2
    lnlike = -0.5 * diff @ INV_COV @ diff
    return jnp.where(jnp.isnan(lnlike), INVALID_LOGLIKE_RAW, lnlike)

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


@jax.jit
def jax_ln_prior_box_batch(thetas, lower, upper):
    in_bounds = jnp.all((thetas >= lower) & (thetas <= upper), axis=1)
    log_norm = -jnp.log(jnp.prod(upper - lower))
    return jnp.where(in_bounds, log_norm, -jnp.inf)


@jax.jit
def jax_likelihood_batch(thetas):
    return jax.vmap(jax_likelihood)(thetas)


@jax.jit
def jax_ln_posterior_batch(thetas, lower, upper):
    lp = jax_ln_prior_box_batch(thetas, lower, upper)
    ll = jax_likelihood_batch(thetas)
    return jnp.where(jnp.isfinite(lp), lp + ll, -jnp.inf)


def ln_posterior_vectorized(theta_batch, lower, upper):
    """Vectorized log-posterior for emcee (shape: nwalkers x ndim)."""
    theta_arr = np.asarray(theta_batch, dtype=np.float64)
    if theta_arr.ndim == 1:
        return float(ln_posterior(theta_arr, lower, upper))

    # Fast path: evaluate the full walker batch in one JAX call.
    try:
        lnps = jax_ln_posterior_batch(
            jnp.asarray(theta_arr),
            jnp.asarray(lower, dtype=jnp.float64),
            jnp.asarray(upper, dtype=jnp.float64),
        )
        return np.asarray(lnps, dtype=np.float64)
    except Exception:
        # Robust fallback for any non-vectorizable corner case.
        return np.array([float(ln_posterior(t, lower, upper)) for t in theta_arr], dtype=np.float64)


def run_small_cosmo_tt(
    nchains=50,
    samples_per_chain=1500,
    nburn=500,
    plot_corner=True,
    vectorize_emcee=True,
):
    plot_dir = "examples/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Parameters from notebook: sigma8, Omega_c, Omega_b, h, n_s
    labels = ['sigma8', 'Omega_c', 'Omega_b', 'h', 'n_s']
    ndim = len(labels)
    # Single global parameter reorder used across all methods in this run.
    # New order: sigma8, Omega_c, h, n_s, Omega_b
    param_order = np.array([0, 1, 3, 4, 2], dtype=int)
    inv_param_order = np.argsort(param_order)
    reorder_tag = "ogreordered_" + "-".join(map(str, param_order.tolist()))
    
    if False:
        approximation_domain = torch.tensor([
            [0.7, 0.9],   # sigma8 (Truth ~0.81)
            [0.2, 0.4],   # Omega_c (Truth ~0.26)
            [0.025, 0.08], # Omega_b (Truth ~0.05)
            [0.45, 0.85],   # h (Truth ~0.67)
            [0.8, 1.1],   # n_s (Truth ~0.96)
        ], dtype=torch.float64)

    approximation_domain = torch.tensor([
        [0.7, 0.9],   # sigma8 (Truth ~0.81)
        [0.23, 0.3],   # Omega_c (Truth ~0.26)
        [0.04, 0.06], # Omega_b (Truth ~0.05)
        [0.64, 0.8],   # h (Truth ~0.67)
        [0.84, 1.10],   # n_s (Truth ~0.96)
    ], dtype=torch.float64)
    approximation_domain = approximation_domain[param_order.tolist(), :]
    labels = [labels[i] for i in param_order]

    # Extract lower and upper bounds from limits
    lower_tt, upper_tt = np.array(list(zip(*approximation_domain.tolist())))
    print("Parameter bounds for TT approximation:")
    for i, label in enumerate(labels):
        print(f"  {label}: [{lower_tt[i]:.3f}, {upper_tt[i]:.3f}]")


    # Box bounds
    lower_prior = np.array([0.5, 0.1, 0.04, 0.64, 0.84])
    upper_prior = np.array([1.2, 0.5, 0.06, 0.82, 1.10])
    lower_prior = lower_prior[param_order]
    upper_prior = upper_prior[param_order]

    def _to_physical_order(theta):
        arr = np.asarray(theta)
        return arr[..., inv_param_order]

    def ln_posterior_reordered(theta, lower, upper):
        return ln_posterior(
            _to_physical_order(theta),
            _to_physical_order(lower),
            _to_physical_order(upper),
        )

    def ln_posterior_vectorized_reordered(theta_batch, lower, upper):
        theta_arr = np.asarray(theta_batch)
        if theta_arr.ndim == 1:
            return float(ln_posterior_reordered(theta_arr, lower, upper))
        return ln_posterior_vectorized(
            _to_physical_order(theta_arr),
            _to_physical_order(lower),
            _to_physical_order(upper),
        )

    # Model parameters for RQSpline
    epochs_num = 80
    temperature = 0.9
    training_proportion = 0.5
    standardize = True
    n_layers = 4
    n_bins = 8
    hidden_size = [64, 64]

    emcee_harmonic = True
    if emcee_harmonic:

        clock = time.process_time()

        # 1. Run Emcee
        print("Run sampling...")
        # Initialize walkers in a small ball around the center of the prior
        pos = lower_prior + (upper_prior - lower_prior) * (0.45 + 0.1 * np.random.rand(nchains, ndim))
        
        if vectorize_emcee:
            # Warm-up compile so the first MCMC step does not pay full JIT cost.
            _ = ln_posterior_vectorized_reordered(pos[: min(8, nchains)], lower_prior, upper_prior)
            print("Using vectorized emcee log-posterior (JAX batched).")
            sampler = emcee.EnsembleSampler(
                nchains,
                ndim,
                ln_posterior_vectorized_reordered,
                args=[lower_prior, upper_prior],
                vectorize=True,
            )
        else:
            print("Using scalar emcee log-posterior.")
            sampler = emcee.EnsembleSampler(
                nchains,
                ndim,
                ln_posterior_reordered,
                args=[lower_prior, upper_prior],
                vectorize=False,
            )
        rstate = np.random.get_state()
        sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate, progress=True)
        
        samples_emcee = np.ascontiguousarray(sampler.chain[:, nburn:, :])
        lnprob_emcee = np.ascontiguousarray(sampler.lnprobability[:, nburn:])

        # Plot results
        hm.utils.plot_getdist(samples_emcee.reshape(-1, ndim), labels=labels)
        plt.savefig(
            f"{plot_dir}/small_cosmo_emcee_corner_{reorder_tag}.png",
            bbox_inches="tight",
            dpi=300,
        )
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
            plt.savefig(
                f"{plot_dir}/small_cosmo_hm_corner_{reorder_tag}.png",
                bbox_inches="tight",
                dpi=300,
            )
        clock = time.process_time() - clock
        hm.logs.info_log("Execution time = {}s".format(clock))
    
    tt_evidence = True
    if tt_evidence:
        theta_ref = 0.5 * (lower_tt + upper_tt)
        lnpost_ref = float(ln_posterior_reordered(theta_ref, lower_tt, upper_tt))

        def neglog_posterior_torch(theta_torch, lower, upper):
            """Compute shifted negative log-posterior for TT (numerical stability)."""
            theta_np = theta_torch.detach().cpu().numpy()
            # Scalar loop for TT evaluation (avoids JAX recompilation)
            lnps = np.array(
                [float(ln_posterior_reordered(t, lower, upper)) for t in theta_np],
                dtype=np.float64,
            )
            # Deterministic penalty for invalid values
            lnps = np.where(np.isfinite(lnps), lnps, INVALID_LOGLIKE_RAW)
            # DIRT expects negative log target, shift by reference point
            neglogps = lnpost_ref - lnps
            return torch.tensor(neglogps, dtype=theta_torch.dtype, device=theta_torch.device)

        # ===========================================================================
        # Configure Tensor Train (deep_tensor)
        # ===========================================================================
        hm.logs.info_log("Building Tensor Train approximation...")

        # Create a partial function with lower and upper bounds pre-specified
        neglog_posterior_torch_partial = partial(neglog_posterior_torch, lower=lower_prior, upper=upper_prior)
        target_func = dt.TargetFunc(neglog_posterior_torch_partial)
        
        reference = dt.UniformReference() 
        preconditioner = dt.UniformMapping(approximation_domain, reference)
        
        tt_max_als = 1
        tt_init_rank = 10
        tt_num_elems = 50
        tt_options = dt.TTOptions(max_als=tt_max_als, init_rank=tt_init_rank, tt_method="fixed_rank")
        basis = dt.Lagrange1(num_elems=tt_num_elems)
        bases = dt.ApproxBases(basis, ndim)
        tt_tag = f"als{tt_max_als}_r{tt_init_rank}_e{tt_num_elems}"

        print(
            f"TT config: max_als={tt_max_als}, init_rank={tt_init_rank}, "
            f"num_elems={tt_num_elems}"
        )

        tt = dt.TT(tt_options)
        ftt = dt.FTT(bases, tt)
        bridge = dt.SingleLayer()
        dirt = dt.DIRT(target_func, preconditioner, ftt, bridge)

        hm.logs.info_log("Generating independent samples from TT...")
        startTime = time.time()
        print(f"Generating random samples from the reference distribution for TT evaluation...")
        num_sampl = nchains * (samples_per_chain-nburn)
        rs = reference.random(n=num_sampl, d=ndim)

        print(f"Evaluating TT at {num_sampl} samples...")
        xs, neglogfxs_sirt = dirt.eval_irt(rs)

        # Evaluate shifted target at TT samples for diagnostics
        neglogfxs_exact = target_func(xs)
        res = dt.run_independence_sampler(xs, neglogfxs_sirt, neglogfxs_exact)
        print(f'Time to generate {num_sampl} TT-approx samples: {(time.time()-startTime):.2f}s')

        print(f"Acceptance rate: {res.acceptance_rate:.3f}")
        if hasattr(res, "iacts") and len(res.iacts) > 0:
            iact_msg = ", ".join([f"{v:.3f}" for v in res.iacts[:min(3, len(res.iacts))]])
            print(f"IACT preview: {iact_msg}")
        
        # Use independence-corrected samples with exact potentials
        samples_torch = res.xs
        samples_tt_np = samples_torch.detach().numpy()

        # Undo shift using exact potentials at corrected samples
        lnprob_tt = lnpost_ref - res.potentials.detach().numpy()

        # Reshape for harmonic (nchains, nsamples, ndim)
        samples = samples_tt_np.reshape(nchains, samples_per_chain - nburn, ndim)

        hm.logs.info_log("Using TT-approximate log-probabilities for Harmonic...")
        lnprob = lnprob_tt.reshape(nchains, samples_per_chain - nburn)

        # ===========================================================================
        # Evidence Calculation (harmonic)
        # ===========================================================================
        hm.logs.info_log("Fitting Harmonic Flow model...")
        chains = hm.Chains(ndim)
        chains.add_chains_3d(samples, lnprob)
        chains_train, chains_test = hm.utils.split_data(chains, training_proportion=0.5)

        model = hm.model.RQSplineModel(
            ndim,
            n_layers=n_layers,
            n_bins=n_bins,
            hidden_size=hidden_size,
            standardize=standardize,
            temperature=temperature,
        )
        model.fit(chains_train.samples, epochs=epochs_num, verbose=True, batch_size=256)

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
            hm.utils.plot_getdist(samples_tt_np, labels=labels)
            plt.savefig(
                f"{plot_dir}/small_cosmo_tt_corner_{tt_tag}_{reorder_tag}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.title(f"Samples from TT approximation ({reorder_tag})")
            plt.show()

            #Plot trained flow vs TT samples
            flow_samples_tt = np.array(model.sample(samples_tt_np.shape[0]))
            hm.utils.plot_getdist_compare(samples_tt_np, flow_samples_tt, labels=labels)
            plt.savefig(
                f"{plot_dir}/small_cosmo_tt_vs_flow_corner_{tt_tag}_{reorder_tag}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.show()
        
        def estimate_evidence(
            neglogpost: Callable[[torch.Tensor], torch.Tensor],
            dirt: dt.DIRT,
            num_samples: int,
        ) -> float:
            """Computes an importance sampling estimate of the log evidence."""

            rs = dirt.reference.random(n=num_samples, d=ndim)
            xs, neglogposts_dirt = dirt.eval_irt(rs)

            # Evaluate same shifted target as TT was built on
            neglogposts_exact = neglogpost(xs)

            # Both are on the same scale; compare directly
            res = dt.run_importance_sampling(neglogposts_dirt, neglogposts_exact)

            # No shift applied, return log norm directly
            return res.log_norm.item()

        clock = time.process_time()

        # Keep a separate importance-sampling evidence estimate.
        num_samples_tt = nchains * (samples_per_chain - nburn)
        log_ev = estimate_evidence(target_func, dirt, num_samples_tt)
        print(f"TT importance sampling ln_evidence: {log_ev:.4f}")

        clock = time.process_time() - clock
        print(f"TT importance sampling evidence estimation completed in {clock:.2f} seconds")

    if tt_evidence and emcee_harmonic:
        print("\nComparison of evidence estimates:")
        print(f"Harmonic + emcee estimate:{ln_evidence_hm:.4f} +/- {-err_ln_inv_evidence_hm[1]} {-err_ln_inv_evidence_hm[0]}")
        print(f"Harmonic + tt posterior samples ln_evidence: {ln_evidence_hm_tt} +/- {-err_ln_inv_evidence_hm_tt[1]} {-err_ln_inv_evidence_hm_tt[0]}")
        print(f" - TT importance sampling estimate: ln_evidence ~ {log_ev:.4f}")


if __name__ == "__main__":
    hm.logs.setup_logging()
    run_small_cosmo_tt(
        nchains=100,
        samples_per_chain=3000,
        nburn=1000,
        vectorize_emcee=True,
    )