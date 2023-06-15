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

def ln_likelihood(x_mean, x_std, x_n, mu, tau):
	"""Compute log_e of likelihood.

	Args:

		x_mean: Mean of simulated data.

		x_std: Standard deviation of simulated data.

		x_n: Number of samples of simulated data.

		mu: Mu value for which to evaluate prior.

		tau: Tau value for which to evaluate prior.

	Returns:

		double: Value of log_e likelihood at specified (mu, tau) point.

	"""

	return -0.5 * x_n * tau * (x_std**2 + (x_mean-mu)**2) \
		- 0.5 * x_n * np.log(2 * np.pi) + 0.5 * x_n * np.log(tau)


def ln_prior(mu, tau, prior_params):
	"""Compute log_e of prior.

	Args:

		mu: Mean value for which to evaluate prior.

		tau: Precision value for which to evaluate prior.

		prior_params: Tuple of prior parameters, including (mu_0, tau_0,
			alpha_0, beta_0).

	Returns:

		double: Value of log_e prior at specified (mu, tau) point.

	"""

	if tau < 0:
		return -np.inf

	mu_0, tau_0, alpha_0, beta_0 = prior_params

	ln_pr = alpha_0 * np.log(beta_0) + 0.5 * np.log(tau_0)
	ln_pr += - sp.gammaln(alpha_0) - 0.5 * np.log(2*np.pi)
	ln_pr += (alpha_0 - 0.5) * np.log(tau)
	ln_pr += -beta_0 * tau
	ln_pr += -0.5 * tau_0 * tau * (mu - mu_0)**2

	return ln_pr


def ln_posterior(theta, x_mean, x_std, x_n, prior_params):
	"""Compute log_e of posterior.

	Args:

		theta: Position (mu, tau) at which to evaluate posterior.

		x_mean: Mean of simulated data.

		x_std: Standard deviation of simulated data.

		x_n: Number of samples of simulated data.

		prior_params: Tuple of prior parameters, including (mu_0, tau_0,
			alpha_0, beta_0).

	Returns:

		double: Value of log_e posterior at specified (mu, tau) point.

	"""

	mu, tau = theta

	ln_pr = ln_prior(mu, tau, prior_params)

	if not np.isfinite(ln_pr):
		return -np.inf

	ln_L = ln_likelihood(x_mean, x_std, x_n, mu, tau)

	return  ln_L + ln_pr


def ln_analytic_evidence(x_mean, x_std, x_n, prior_params):
	"""Compute analytic evidence.

	Args:

		x_mean: Mean of simulated data.

		x_std: Standard deviation of simulated data.

		x_n: Number of samples of simulated data.

		prior_params: Tuple of prior parameters, including (mu_0, tau_0,
			alpha_0, beta_0).

	Returns:

		double: Value of log_e evidence computed analytically.

	"""

	mu_0, tau_0, alpha_0, beta_0 = prior_params

	tau_n  = tau_0  + x_n
	alpha_n = alpha_0 + x_n/2
	beta_n  = beta_0 + 0.5 * x_n * x_std**2 \
		+ tau_0 * x_n * (x_mean - mu_0)**2 / (2 * (tau_0 + x_n))

	ln_z  = sp.gammaln(alpha_n) - sp.gammaln(alpha_0)
	ln_z += alpha_0 * np.log(beta_0) - alpha_n * np.log(beta_n)
	ln_z += 0.5 * np.log(tau_0) - 0.5 * np.log(tau_n)
	ln_z -= 0.5 * x_n * np.log(2*np.pi)

	return ln_z


def run_example(ndim=2, nchains=100, samples_per_chain=1000,
				nburn=500, plot_corner=False, plot_surface=False,
				plot_comparison=False):
	"""Run Normal-Gamma example.

	Args:

		ndim: Dimension.

		nchains: Number of chains.

		samples_per_chain: Number of samples per chain.

		nburn: Number of burn in samples for each chain.

		plot_corner: Plot marginalised distributions if true.

		plot_surface: Plot surface and samples if true.

		plot_comparison: Plot accuracy for various tau priors if true.

	"""

	if ndim != 2:
		raise ValueError("Only ndim=2 is supported (ndim={} specified)"
			.format(ndim))

	#===========================================================================
	# Configure Parameters.
	#===========================================================================
	"""
	Configure machine learning parameters
	"""
	n_meas = 100
	mu_in  = 0.0
	tau_in = 1.0
	tau_array = [1E-4, 1E-3, 1E-2, 1E-1, 1E0]

	savefigs = True
	created_plots = False


	training_proportion = 0.5
	var_scale = 0.8
	epochs_num = 30

	#===========================================================================
	# Simulate data 
	#===========================================================================
	hm.logs.info_log('Simulate data...')
	"""
	Construct simulations of data one might observe from a typical normal-gamma
	model.
	"""
	x = np.random.normal(mu_in, np.sqrt(1/tau_in), (n_meas))
	x_mean = np.mean(x)
	x_std = np.std(x)
	x_n = x.size
	hm.logs.debug_log('x: mean = {}, std = {}, n = {}'.format(x_mean, x_std, x_n))

	summary = np.empty((len(tau_array), 4), dtype=float)

	# Start timer.
	clock = time.process_time()

	#===========================================================================
	# Loop over all values of Tau one wishes to consider
	#===========================================================================
	"""
	Run many realisations for each Tau value.
	"""
	for i_tau, tau_prior in enumerate(tau_array):
		
		hm.logs.info_log('Considering tau = {}...'.format(tau_prior))

		prior_params = (0.0, tau_prior, 1E-3, 1E-3)

		#===================================================================
		# Run Emcee to recover posterior samples 
		#===================================================================
		hm.logs.info_log('Run sampling...')
		"""
		Feed emcee the ln_posterior function, starting positions and recover 
		chains.
		"""
		pos = [np.array([x_mean, 1.0/x_std**2]) \
			   + x_std * np.random.randn(ndim) / np.sqrt(x_n) \
			   for i in range(nchains)]
		sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, \
			args=(x_mean, x_std, x_n, prior_params))
		rstate = np.random.get_state()
		sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
		samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
		lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

		#===================================================================
		# Configure emcee chains for harmonic
		#===================================================================
		hm.logs.info_log('Configure chains...')
		"""
		Configure chains for the cross-validation stage.
		"""
		chains = hm.Chains(ndim)
		chains.add_chains_3d(samples, lnprob)
		chains_train, chains_test = hm.utils.split_data(chains, \
			training_proportion=training_proportion)

        #=======================================================================
        # Fit model
        #=======================================================================
		hm.logs.info_log('Fit model for {} epochs...'.format(epochs_num))
		model = model_nf.RealNVPModel(ndim)
		model.fit(chains_train.samples, chains_train.ln_posterior, epochs=epochs_num) 

		#===================================================================
		# Computing evidence using learnt model and emcee chains
		#===================================================================
		hm.logs.info_log('Compute evidence...')
		"""
		Instantiates the evidence class with a given model. Adds some chains 
		and computes the log-space evidence (marginal likelihood).
		"""
		ev = hm.Evidence(chains_test.nchains, model)
		ev.add_chains(chains_test, bulk_calc=True, var_scale=var_scale)
		ln_evidence, ln_evidence_std = ev.compute_ln_evidence()

		# Compute analytic evidence.
		ln_evidence_analytic = ln_analytic_evidence(x_mean, \
			x_std, x_n, prior_params)
		evidence_analytic = np.exp(ln_evidence_analytic)

		# Collate values.
		summary[i_tau, 0] = tau_prior
		summary[i_tau, 1] = ln_evidence_analytic
		summary[i_tau, 2] = ln_evidence
		summary[i_tau, 3] = ln_evidence_std

		# ==================================================================
		# Display evidence computation results.
		# ==================================================================
		hm.logs.info_log('Evidence: analytic = {}, estimated = {}'
			.format(evidence_analytic, np.exp(ln_evidence)))
		hm.logs.info_log('Evidence: std = {}, std / estimated = {}'
			.format(np.exp(ln_evidence_std), \
					np.exp(ln_evidence_std - ln_evidence)))
		diff = np.log(np.abs(evidence_analytic - np.exp(ln_evidence)))
		hm.logs.info_log('Evidence: |analytic - estimated| / estimated = {}'
			.format(np.exp(diff - ln_evidence)))

		# ==================================================================
		# Display logarithmic evidence computation results.
		# ==================================================================
		hm.logs.debug_log('Ln Evidence: analytic = {}, estimated = {}'
			.format(ln_evidence_analytic, ln_evidence))
		diff = np.abs(ln_evidence_analytic - ln_evidence)
		hm.logs.debug_log('Ln Evidence: |analytic - estimated| / estimated = {}'
						  .format(diff/ln_evidence))

		#===========================================================================
		# Display more technical details
		#===========================================================================
		hm.logs.debug_log('---------------------------------')
		hm.logs.debug_log('Technical Details')
		hm.logs.debug_log('---------------------------------')
		hm.logs.debug_log('lnargmax = {}, lnargmin = {}'
			.format(ev.lnargmax, ev.lnargmin))
		hm.logs.debug_log('lnprobmax = {}, lnprobmin = {}'
			.format(ev.lnprobmax, ev.lnprobmin))
		hm.logs.debug_log('lnpredictmax = {}, lnpredictmin = {}'
			.format(ev.lnpredictmax, ev.lnpredictmin))
		hm.logs.debug_log('---------------------------------')
		hm.logs.debug_log('shift = {}, shift setting = {}'
			.format(ev.shift_value, ev.shift))
		hm.logs.debug_log('running sum total = {}'
			.format(sum(ev.running_sum)))
		hm.logs.debug_log('running sum = \n{}'
			.format(ev.running_sum))
		hm.logs.debug_log('nsamples per chain = \n{}'
			.format(ev.nsamples_per_chain))
		hm.logs.debug_log('nsamples eff per chain = \n{}'
			.format(ev.nsamples_eff_per_chain))
		hm.logs.debug_log('===============================')

		# Create corner/triangle plot.
		if plot_corner:

			labels = [r'$\mu$', r'$\tau$']
			utils.plot_corner(samples.reshape((-1, ndim)), labels)
			if savefigs:
				plt.savefig('examples/plots/nvp_normalgamma_corner_tau' +
							str(tau_prior) +
							'.pdf',
							bbox_inches='tight')

			labels = [r'\mu', r'\tau']
			utils.plot_getdist(samples.reshape((-1, ndim)), labels)
			if savefigs:
				plt.savefig('examples/plots/nvp_normalgamma_getdist_tau' +
							str(tau_prior) +
							'.pdf',
							bbox_inches='tight')

			plt.show(block=False)

			#=======================================================================
            # Visualise distributions
            #=======================================================================

			num_samp = chains_train.samples.shape[0]
			samps_compressed = np.array(model.sample(num_samp, var_scale=var_scale))

			utils.plot_getdist_compare(chains_train.samples, samps_compressed, labels)
	
			if savefigs:
				plt.savefig('examples/plots/nvp_normalgamma_corner_all.png',
                                bbox_inches='tight')
		
			created_plots = True

		if plot_surface:

			# Evaluate posterior on grid.
			ln_posterior_func = partial(ln_posterior,
				x_mean=x_mean, x_std=x_std, x_n=x_n,
				prior_params=prior_params)
			ln_posterior_grid, x_grid, y_grid = \
				utils.eval_func_on_grid(ln_posterior_func,
										xmin=-0.6, xmax=0.6,
										ymin=0.4, ymax=1.8,
										nx=500, ny=500)

			# Plot posterior image.
			ax = utils.plot_image(np.exp(ln_posterior_grid),
								  x_grid, y_grid,
								  samples=None,
								  #samples.reshape((-1,ndim)),
								  colorbar_label=r'$\mathcal{L}$')
			# ax.set_clim(vmin=0.0, vmax=0.003)
			if savefigs:
				plt.savefig('examples/plots/' +
							'nvp_normalgamma_posterior_image' +
							str(tau_prior) + '.png',
							bbox_inches='tight')

			# Evaluate model on grid.
			model_grid, x_grid, y_grid = \
				utils.eval_func_on_grid(model.predict,
										xmin=-0.6, xmax=0.6,
										ymin=0.4, ymax=1.8,
										nx=500, ny=500)

			# Plot model.
			ax = utils.plot_image(model_grid, x_grid, y_grid,
				colorbar_label=r'$\log \varphi$')
			# ax.set_clim(vmin=-2.0, vmax=2.0)
			if savefigs:
				plt.savefig('examples/plots/' +
							'nvp_normalgamma_model_image' +
							str(tau_prior) +
							'.png',
							bbox_inches='tight')

			# Plot exponential of model.
			ax = utils.plot_image(np.exp(model_grid),
								  x_grid, y_grid,
								  colorbar_label=r'$\varphi$')
			# ax.set_clim(vmin=0.0, vmax=10.0)
			if savefigs:
				plt.savefig('examples/plots/' +
							'nvp_normalgamma_modelexp_image' +
							str(tau_prior) +
							'.png',
							bbox_inches='tight')

	# Display summary results.
	hm.logs.info_log('tau_prior | ln_evidence_analytic | ln_evidence =')
	hm.logs.info_log('{}'.format(summary[:,:-1]))

	# Plot evidence values for different tau priors.
	if plot_comparison:
		created_plots = True

		plt.rcParams.update({'font.size': 15})
		fig, ax = plt.subplots()
		ax.plot(np.array([1E-5, 1E1]), np.ones(2), 'r', linewidth=2)
		ax.set_xlim([1E-5, 1E1])
		ax.set_ylim([0.990, 1.010])
		ax.set_xscale("log")
		ax.set_xlabel(r"Prior size ($\tau_0$)")
		ax.set_ylabel(r"Relative accuracy ($z_{\rm estimated}/z_{\rm analytic}$)")
		ax.errorbar(tau_array, np.exp(summary[:,2])/np.exp(summary[:,1]),
			yerr=np.exp(summary[:,3])/np.exp(summary[:,1]),
			fmt='b.', capsize=4, capthick=2, elinewidth=2)
		if savefigs:
			plt.savefig('examples/plots/nvp_normalgamma_comparison.pdf',
						bbox_inches='tight')
		plt.show(block=False)

	#===========================================================================
	# End Timer.
	clock = time.process_time() - clock
	hm.logs.info_log('execution_time = {}s'.format(clock))


	if created_plots:
		input("\nPress Enter to continue...")


if __name__ == '__main__':

	# Setup logging config.
	hm.logs.setup_logging()

	# Define parameters.
	ndim = 2 # Only 2 dimensional case supported.
	nchains = 200
	samples_per_chain = 1500
	nburn = 500
	np.random.seed(1)

	hm.logs.info_log('Normal-Gamma example') 

	hm.logs.debug_log('-- Selected Parameters --')

	hm.logs.debug_log('Dimensionality = {}'.format(ndim))
	hm.logs.debug_log('Number of chains = {}'.format(nchains))
	hm.logs.debug_log('Samples per chain = {}'.format(samples_per_chain))
	hm.logs.debug_log('Burn in = {}'.format(nburn))
	
	hm.logs.debug_log('-------------------------')

	# Run example.
	samples = run_example(ndim, nchains, samples_per_chain, nburn,
						  plot_corner=True, plot_surface=False,
						  plot_comparison=True)