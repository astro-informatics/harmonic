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

# Setup Logging config
hm.logs.setup_logging()

def ln_likelihood(y, theta, x):
	"""
	Compute log_e of Pima Indian likelihood

	Args: 
	    - y: 
	        Vector of incidence. 1=diabetes, 0=no diabetes
	    - theta: 
	        Vector of parameter variables associated with covariates x.
	    - x: 
	        Vector of data covariates (e.g. NP, PGC, BP, TST, DMI e.t.c.).
	       
	    
	Returns:
	    - double: 
	        Value of log_e likelihood at specified point in parameter space.
	"""
	ln_p = compute_ln_p(theta, x)
	ln_pp = x.dot(theta) - ln_p
	return y.T.dot(ln_p) + (1-y).T.dot(ln_pp)

def ln_prior(tau, theta): 
	"""
	Compute log_e of Pima Indian multivariate gaussian prior

	Args: 
	    - tau: 
	        Characteristic width of posterior \in [0.01,1]
	    - theta: 
	        Vector of parameter variables associated with covariates x.
	       
	Returns:
	    - double: 
	        Value of log_e prior at specified point in parameter space.
	"""
	d = len(theta)
	return 0.5 * d * np.log(tau/(2.*np.pi)) \
		 - 0.5 * tau * theta.T.dot(theta)

def ln_posterior(theta, tau, x, y): 
	"""
	Compute log_e of Pima Indian multivariate gaussian prior

	Args: 
		- theta: 
	        Vector of parameter variables associated with covariates x.
	    - tau: 
	        Characteristic width of posterior \in [0.01,1]
	    - x: 
	        Vector of data covariates (e.g. NP, PGC, BP, TST, DMI e.t.c.).
	    - y: 
	        Vector of incidence. 1=diabetes, 0=no diabetes
	       
	Returns:
	    - double: 
	        Value of log_e posterior at specified point in parameter space.
	"""
	return ln_prior(tau,theta) + ln_likelihood(y, theta, x)

def compute_ln_p(theta, x):
	"""
	Computes log_e probability ln(p) to be used in likelihood function

	Args: 
	    - theta: 
	        Vector of parameter variables associated with covariates x.
	    - x:
	    	Vector of data covariates (e.g. NP, PGC, BP, TST, DMI e.t.c.).
	       
	Returns:
	    - Ln(p):
			Vector of the log-probabilities p to use in likelihood.
	"""
	return - np.log(1.0 + np.exp(x.dot(theta)))


def run_example(ndim=8, nchains=100, samples_per_chain=1000, 
                nburn=500, verbose=True, 
                plot_corner=False, plot_surface=False,
                plot_comparison=False):

	hm.logs.low_log('---------------------------------')
	hm.logs.high_log('Pima Indian example')
	hm.logs.high_log('Dimensionality = {}'.format(ndim))
	hm.logs.low_log('---------------------------------')

	if ndim != 8:
	    raise ValueError("Only ndim=8 is supported (ndim={} specified)"
	        .format(ndim))


	#===========================================================================
	# Load Pima Indian data.
	#===========================================================================
	hm.logs.high_log('Loading data ...')
	hm.logs.low_log('---------------------------------')
	"""
	https://gist.github.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f
	"""
	data = np.loadtxt('./data/Pima_Indian.dat')

	"""
	 x[:,0] = Number of times pregnant.
	 x[:,1] = Plasma glucose concentration a 2 hours in an oral glucose 
	 		  tolerance test.
	 x[:,2] = Diastolic blood pressure (mm Hg).
	 x[:,3] = Triceps skin fold thickness (mm).
	 x[:,4] = 2-Hour serum insulin (mu U/ml).
	 x[:,5] = Body mass index (weight in kg/(height in m)^2).
	 x[:,6] = Diabetes pedigree function.
	 x[:,7] = Age (years).
	"""
	x=np.zeros((len(data), ndim))
	for i in range(ndim):
		x[:,i] = data[:,i] - np.nanmean(data[:,i])


	"""
	y[:] = 1 if patient has diabetes, 0 if patient does not have diabetes.
	"""
	y = data[:,ndim]

	"""
	Configure some general parameters.
	Tau should be varied in [0.01, 1] for testing.
	"""
	tau = 0.01    
	savefigs = True

	"""
	Configure machine learning parameters
	"""
	nfold = 3
	training_proportion = 0.25
	hyper_parameters_MGMM = [[1, 1E-8, 0.1, 6, 10], [2, 1E-8, 0.5, 6, 10]]
	hyper_parameters_sphere = [None]
	domains_sphere = [np.array([1E-1,5E0])]
	domains_MGMM = [np.array([1E-1,5E0])]

	#===========================================================================
	# Compute random positions to draw from for emcee sampler.
	#===========================================================================
	"""
	Initial positions for each chain for each covariate \in [0,8).
	Simply drawn from directly from each covariate prior.
	"""
	pos_0 = (0.5*tau**10)*np.random.randn(nchains)
	pos_1 = (0.5*tau**10)*np.random.randn(nchains)
	pos_2 = (0.5*tau**10)*np.random.randn(nchains)
	pos_3 = (0.5*tau**10)*np.random.randn(nchains)
	pos_4 = (0.5*tau**10)*np.random.randn(nchains)
	pos_5 = (0.5*tau**10)*np.random.randn(nchains)
	pos_6 = (0.5*tau**10)*np.random.randn(nchains)
	pos_7 = (0.5*tau**10)*np.random.randn(nchains)

	"""
	Concatenate these positions into a single variable 'pos'.
	"""
	pos = np.c_[pos_0, pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7]

	# Start Timer.
	clock = time.clock()

	#===========================================================================
	# Run Emcee to recover posterior sampels 
	#===========================================================================
	hm.logs.high_log('Run sampling...')
	hm.logs.low_log('---------------------------------')
	"""
	Feed emcee the ln_posterior function, starting positions and recover chains.
	"""
	sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, \
		                            args=(tau, x, y))
	rstate = np.random.get_state()
	sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
	samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
	lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

	#===========================================================================
	# Configure emcee chains for harmonic
	#===========================================================================
	hm.logs.high_log('Configuring chains...')
	hm.logs.low_log('---------------------------------')
	"""
	Configure chains for the cross validation stage.
	"""
	chains = hm.Chains(ndim)
	chains.add_chains_3d(samples, lnprob)
	chains_train, chains_test = hm.utils.split_data(chains, \
	    training_proportion=training_proportion)

	#===========================================================================
	# Perform cross-validation
	#===========================================================================
	hm.logs.high_log('Perform cross-validation...')
	hm.logs.low_log('---------------------------------')
	"""
	There are several different machine learning models. Cross-validation
	allows the software to select the optimal model and the optimal model 
	hyper-parameters for a given situation.
	"""
	# MGMM model
	validation_variances_MGMM = \
	    hm.utils.cross_validation(chains_train, 
	        domains_MGMM, \
	        hyper_parameters_MGMM, \
	        nfold=nfold, 
	        modelClass=hm.model.ModifiedGaussianMixtureModel, \
	        verbose=verbose, seed=0)                
	hm.logs.low_log('Validation variances of MGMM = {}'
		.format(validation_variances_MGMM))
	best_hyper_param_MGMM_ind = np.argmin(validation_variances_MGMM)
	best_hyper_param_MGMM = \
	    hyper_parameters_MGMM[best_hyper_param_MGMM_ind]

	# Hyper-spherical model
	validation_variances_sphere = \
	    hm.utils.cross_validation(chains_train, 
	        domains_sphere, \
	        hyper_parameters_sphere, nfold=nfold, 
	        modelClass=hm.model.HyperSphere, 
	        verbose=verbose, seed=0)
	hm.logs.low_log('Validation variances of sphere = {}'
	    .format(validation_variances_sphere))
	best_hyper_param_sphere_ind = np.argmin(validation_variances_sphere)
	best_hyper_param_sphere = \
	    hyper_parameters_sphere[best_hyper_param_sphere_ind]

	#===========================================================================
	# Select the optimal model from cross-validation results
	#===========================================================================
	hm.logs.high_log('Select optimal model...')
	hm.logs.low_log('---------------------------------')
	"""
	This simply uses the cross-validation results to choose the model which 
	has the smallest validation variance -- i.e. the best model for the job.
	"""
	best_var_MGMM = \
	    validation_variances_MGMM[best_hyper_param_MGMM_ind]
	best_var_sphere = \
	    validation_variances_sphere[best_hyper_param_sphere_ind]
	if best_var_MGMM < best_var_sphere:            
	    hm.logs.high_log('Using MGMM with hyper_parameters = {}'
	        .format(best_hyper_param_MGMM))                
	    model = hm.model.ModifiedGaussianMixtureModel(ndim, \
	        domains_MGMM, hyper_parameters=best_hyper_param_MGMM)
	    model.verbose=False
	else:
	    hm.logs.high_log('Using HyperSphere')
	    model = hm.model.HyperSphere(ndim, domains_sphere, \
	            hyper_parameters=best_hyper_param_sphere)
	    model = hm.model.HyperSphere(ndim, domains_sphere,hyper_parameters=None)            

	#===========================================================================
	# Fit learnt model for container function 
	#===========================================================================
	"""
	Once the model is selected the model is fit to chain samples.
	"""
	fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)
	hm.logs.low_log('Fit success = {}'.format(fit_success)) 

	#===========================================================================
	# Computing evidence using learnt model and emcee chains
	#===========================================================================
	hm.logs.high_log('Compute evidence...')
	hm.logs.low_log('---------------------------------')
	"""
	Instantiates the evidence class with a given model. Adds some chains and 
	computes the log-space evidence (marginal likelihood).
	"""
	ev = hm.Evidence(chains_test.nchains, model)
	ev.add_chains(chains_test)
	# ln_evidence, ln_evidence_std = ev.compute_ln_evidence()

	#===========================================================================
	# End Timer.
	clock = time.clock() - clock
	hm.logs.high_log('Execution time = {}s'.format(clock))
	hm.logs.low_log('---------------------------------')

	# #===========================================================================
	# # Display evidence results 
	# #===========================================================================
	# hm.logs.high_log('Evidence results')
	# hm.logs.low_log('---------------------------------')
	# hm.logs.low_log('ln_evidence) = {}, -np.log(ev.evidence_inv) = {}'
	#     .format(ln_evidence, -np.log(ev.evidence_inv))) 
	# hm.logs.low_log('np.exp(evidence) = {}'.format(np.exp(ln_evidence)))
	# hm.logs.low_log('evidence_std = {}, evidence_std / evidence = {}'
	#     .format(np.exp(ln_evidence_std), np.exp(ln_evidence_std - ln_evidence)))
	#  #===========================================================================
	# # Display evidence results 
	# #===========================================================================
	# hm.logs.high_log('Inverse evidence results')
	# hm.logs.low_log('---------------------------------')
	# hm.logs.low_log('evidence_inv = {}'
	#     .format(ev.evidence_inv))
	# hm.logs.low_log('evidence_inv_std = {}, evidence_inv_std/evidence_inv = {}'
	#     .format(np.sqrt(ev.evidence_inv_var), \
	#             np.sqrt(ev.evidence_inv_var)/ev.evidence_inv))
	# hm.logs.low_log('kurtosis = {}, sqrt( 2 / ( n_eff - 1 ) ) = {}'
	#     .format(ev.kurtosis, np.sqrt(2.0/(ev.n_eff-1))))    
	# hm.logs.low_log('sqrt(ev.evidence_inv_var_var)/ev.evidence_inv_var = {}'
	#     .format(np.sqrt(ev.evidence_inv_var_var)/ev.evidence_inv_var))
	#===========================================================================
	# LOG-SPACE TESTING
	#===========================================================================
	hm.logs.low_log('---------------------------------')
	hm.logs.low_log('ev_var_var test = {}'.format(ev.evidence_inv_var_var))

	hm.logs.high_log('START TESTING LOG-SPACE!')
	hm.logs.low_log('---------------------------------')
	hm.logs.low_log('ln( evidence_inv ) = {}'
	    .format(ev.ln_evidence_inv))
	hm.logs.low_log('ln( evidence_inv_std ) = {}, \
	        ln ( evidence_inv_std/evidence_inv ) = {}'
	    .format(0.5*ev.ln_evidence_inv_var, \
	            0.5 * ev.ln_evidence_inv_var - ev.ln_evidence_inv))
	hm.logs.low_log('ln( kurtosis ) = {}, sqrt( 2 / ( n_eff - 1 ) ) = {}'
	    .format(ev.ln_kurtosis, np.sqrt(2.0/(ev.n_eff-1))))    
	hm.logs.low_log('ln( ev.evidence_inv_var_std/ev.evidence_inv_var ) = {}'
	    .format(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var) )
	hm.logs.low_log('---------------------------------')
	hm.logs.low_log('exp( ln( evidence_inv ) ) = {}'
	    .format(np.exp(ev.ln_evidence_inv) ) )
	hm.logs.low_log('exp( ln( evidence_inv_std ) )= {}, \
	        exp( ln ( evidence_inv_std/evidence_inv ) ) = {}'
	    .format(np.exp( 0.5*ev.ln_evidence_inv_var), \
	            np.exp(0.5 * ev.ln_evidence_inv_var - ev.ln_evidence_inv)) )
	hm.logs.low_log('exp( ln( kurtosis ) ) = {}, sqrt( 2 / ( n_eff - 1 ) ) = {}'
	    .format(np.exp( ev.ln_kurtosis ), np.sqrt(2.0/(ev.n_eff-1))))    
	hm.logs.low_log('exp(ln(ev.evidence_inv_var_std/ev.evidence_inv_var)) = {}'
	    .format(np.exp( 0.5 * ev.ln_evidence_inv_var_var \
	    	                - ev.ln_evidence_inv_var) ) )
	hm.logs.high_log('END TESTING LOG-SPACE!')
	hm.logs.low_log('---------------------------------')
	#===========================================================================
	# Display more technical details
	#===========================================================================
	hm.logs.low_log('---------------------------------')
	hm.logs.low_log('lnargmax = {}, lnargmin = {}'
	    .format(ev.lnargmax, ev.lnargmin))
	hm.logs.low_log('lnprobmax = {}, lnprobmin = {}'
	    .format(ev.lnprobmax, ev.lnprobmin))
	hm.logs.low_log('lnpredictmax = {}, lnpredictmin = {}'
	    .format(ev.lnpredictmax, ev.lnpredictmin))
	hm.logs.low_log('---------------------------------')
	hm.logs.low_log('mean shift = {}, max shift = {}'
	    .format(ev.mean_shift, ev.max_shift))
	hm.logs.low_log('running sum total = {}'
	    .format(sum(ev.running_sum)))
	hm.logs.low_log('running sum = \n{}'
	    .format(ev.running_sum))
	hm.logs.low_log('nsamples per chain = \n{}'
	    .format(ev.nsamples_per_chain))
	hm.logs.low_log('nsamples eff per chain = \n{}'
	    .format(ev.nsamples_eff_per_chain))
	hm.logs.low_log('===============================')


if __name__ == '__main__':
    
    # Define parameters.
    ndim = 8 # Only 8 dimensional case supported (covariate dim(x_i) = 8)
    nchains = 200
    samples_per_chain = 5000
    nburn = 1000
    np.random.seed(3)
    
    # Run example.
    samples = run_example(ndim, nchains, samples_per_chain, nburn, 
                          plot_corner=False, plot_surface=False,
                          plot_comparison=False, 
                          verbose=False)



