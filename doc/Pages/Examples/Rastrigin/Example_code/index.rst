**************************
Code Analysis
**************************
The `Rastrigin function  <https://www.sfu.ca/~ssurjano/rastr.html>`_ is a particularly tricky function often used for benchmarking of algorithms. The functional form is given by

.. math:: f(x) = 10 d + \sum_{i=1}^{d} \bigg [ x_i^2 - 10 \cos ( 2 \pi x_i ) \bigg ]

where :math:`d` is the dimension of the function and the input domain is taken in this case to be :math:`x_i \in [-6.0, 6.0] \: \; \forall i = 1, \dots, d`. The Rastrigin function is particularly pathological as it is highly multimodal however the maxima locations are regularly distributed which simplifies the problem somewhat. 

.. note:: by definition the global minimum is trivially given by :math:`f(x^{\text{min}}) = 0, \: \text{at} \: x^{\text{min}} = (0,\dots,0)`.


Log-Likelihood, log-Prior and log-Posterior
==========================

The log-likelihood function is given by

.. code-block:: python

   def ln_likelihood(x):
    	ndim = x.size
    	f = 10.0 * ndim
    	for i_dim in range(ndim):
        	f += x[i_dim]**2 - 10.0 * np.cos( 2.0 * np.pi * x[i_dim] )
    	return -f

where the rastrigin function :math:`f(x)` is inverted to :math:`-f(x)` so as to form a sensible likelihood function -- *i.e.* a function which converges to a global maximum rather than a global minimum. In the example at hand we assume a simple uniform log-prior defined such that,

.. code-block:: python

   def ln_prior_uniform(x, xmin=-6.0, xmax=6.0, ymin=-6.0, ymax=6.0):
    	if x[0] >= xmin and x[0] <= xmax and x[1] >= ymin and x[1] <= ymax:        
        	return 1.0 / ( (xmax - xmin) * (ymax - ymin) )
    	else:
        	return 0.0

where the prior is uniform over :math:`x_i \in [-6.0, 6.0] \: \; \forall i = 1, \dots, d`. In practice however this prior need not necessarily be uniform. 

.. warning:: One should note that for :math:`d \gg 1` uniform priors quickly become very informative, and as such often constitute poor choices. 

Finally, combining the log-likelihood and log-prior functions we can define the log-posterior function simply by

.. code-block:: python
	
   def ln_posterior(x, ln_prior):
    	ln_L = ln_likelihood(x)
    	if not np.isfinite(ln_L):
        	return -np.inf
    	else:
        	return ln_prior(x) + ln_L
	

MCMC Sampling
==========================
The first step of our evidence computation requires recovering a relatively small number of samples from the given posterior. This can be done in whatever way the user wishes, the only requirement being that a set of chains each with associated samples is provided for subsequent steps.
In our examples we choose to use the excellent `emcee  <http://dfm.io/emcee/current/>`_ python package. Utilizing emcee this example recovers samples via 

.. code-block:: python

   pos = np.random.rand(ndim * nchains).reshape((nchains, ndim)) * 0.5    
   sampler = emcee.EnsembleSampler(nchains, ndim, ln_posteriargs=[ln_prior])
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





