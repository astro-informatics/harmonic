**************************
Code Analysis
**************************
The `Himmelblau function  <http://benchmarkfcns.xyz/benchmarkfcns/himmelblaufcn.html>`_ is a strictly 2-dimensional pathological function often used for benchmarking of algorithms. The functional form is given by

.. math:: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

where the input domain is typically taken to be :math:`x,y \in [-5.0, 5.0]`. The Himmelblau function is difficult in the sense that it is non-convex and multimodal.

.. note:: by definition the global minima are not so trivially given by :math:`f(x^{\text{min}}) = 0, \: \text{at} \: x^{\text{min}} = (3,2),(−2.805118,3.283186),(-3.779310, -3.283186),(3.584458,-1.848126)`.


Log-Likelihood, Log-Prior and Log-Posterior
==========================

The log-likelihood function is given by

.. code-block:: python

   def ln_likelihood(x):
    	f = (x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2
    	return -f

where the rastrigin function :math:`f(x)` is inverted to :math:`-f(x)` so as to form a sensible likelihood function -- *i.e.* a function which converges to a global maximum rather than a global minimum. In the example at hand we assume a simple uniform log-prior defined such that,

.. code-block:: python

   def ln_prior_uniform(x, xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0):
    	if x[0] >= xmin and x[0] <= xmax and x[1] >= ymin and x[1] <= ymax:        
        	return 1.0 / ( (xmax - xmin) * (ymax - ymin) )
    	else:
        	return 0.0

where the log-prior is uniform over :math:`x,y \in [-5.0, 5.0]`.

.. warning:: One should note that for :math:`d \gg 1` uniform priors quickly become very informative, and as such often constitute poor choices -- or should at the very least be chosen carefully.


Finally, combining the log-likelihood and log-prior functions we can define the log-posterior function simply by

.. code-block:: python
	
   def ln_posterior(x, xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0):
    	ln_L = ln_likelihood(x)
    	if not np.isfinite(ln_L):
        	return -np.inf
    	else:
        	return ln_prior(x, xmin, xmax, ymin, ymax) + ln_L
	

MCMC Sampling
==========================
The first step of our evidence computation requires recovering a relatively small number of samples from the given posterior. This can be done in whatever way the user wishes, the only requirement being that a set of chains each with associated samples is provided for subsequent steps.
In our examples we choose to use the excellent `emcee  <http://dfm.io/emcee/current/>`_ python package. Utilizing emcee this example recovers samples via 

.. code-block:: python

   pos = np.random.rand(ndim * nchains).reshape((nchains, ndim)) * 10.0 - 5.0
   sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=[xmin, xmax, ymin, ymax])
   rstate = np.random.get_state()
   sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
   samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
   lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

where the initial positions are drawn randomly from a uniform area of size representative of the region over which the posterior has large support.

Cross-Validation 
==========================
The cross validation step allows Harmonic to copute the optimal hyoer-parameter configuration for a certain class of model for a given set of posterior samples.

There are two main stages to this cross-validation process. First the MCMC chains (in this case from emcee) are configured

.. code-block:: python

   chains = hm.Chains(ndim)
   chains.add_chains_3d(samples, lnprob)
   chains_train, chains_test = hm.utils.split_data(chains, training_proportion=0.5)

before being used as training data to train a network to predict optimal configurations of the hyper-parameters associated with the model class. This is done by

.. code-block:: python

   validation_variances = 
	            hm.utils.cross_validation(chains_train, 
	                                      domain, 
	                                      hyper_parameters, 
	                                      nfold=nfold, 
	                                      modelClass=hm.model.KernelDensityEstimate, 
	                                      verbose=verbose, 
	                                      seed=0)
   best_hyper_param_ind = np.argmin(validation_variances)
   best_hyper_param = hyper_parameters[best_hyper_param_ind]

In this case we choose to used the Kernel Density Estimate (KDE) though others could be selected at this stage with ease.

Evidence estimation
==========================

Finally the now sucessfully trained network is used to make a prediction (fit) the optimal (learnt) container function :math:`\psi` -- *i.e.* the optimal hyper-parameter configuration -- by

.. code-block:: python

   model = hm.model.KernelDensityEstimate(ndim, domain, hyper_parameters=best_hyper_param)
   fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)

This container function is then used with the harmonic mean estimator to construct a robust computation of the Bayesian evidence by

.. code-block:: python

   ev = hm.Evidence(chains_test.nchains, model)    
   ev.add_chains(chains_test)
   ln_evidence, ln_evidence_std = ev.compute_ln_evidence()