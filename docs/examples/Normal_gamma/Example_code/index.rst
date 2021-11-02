The Normal-Gamma distirbution is an interesting example as it is one for which the original harmonic mean estimator catastrophically failed. Further it is claimed that the harmonic mean evidence estimator is insensitive to the prior and should be avoided -- a question addressed within this example. The Normal-Gamma posterior configuration is composed of a likelihood defined as

.. math:: \mathcal{L}(\mathbf{x}|\mu,\tau) = \prod_{i=1}^n p(x_i \vert \mu, \tau)
		  = \frac{\tau^{n/2}}{(2\pi)^{n/2}} e^{-\frac{\tau}{2}\sum_{i=1}^n(x_i-\mu)^2}
		  =\frac{\tau^{n/2}}{(2\pi)^{n/2}} e^{-\frac{\tau}{2} n (s^2 + (\bar{x} - \mu)^2)}

and a prior defined as

.. math:: \pi(\mu, \tau) = \frac{{\beta_0}^{\alpha_0} 		\sqrt{\tau_0}}{\Gamma(\alpha_0)\sqrt{2\pi}}\tau^{\alpha_0-1/2} e^{-\beta_0\tau}
		  e^{-\frac{\tau_0\tau(\mu-\mu_0)^2}{2}}

where :math:`\alpha_0, \beta_0, \tau_0` and :math:`\mu_0` are parameters that define the prior and :math:`n` is the number of data points :math:`x_i` we have. 
A useful property of this particular posterior is that it permits a closed for analytic expression for the evidence, given as

.. math:: z = \frac{\Gamma(\alpha_n)}{\Gamma(\alpha_0)}\frac{\beta^\alpha_0}{\beta_n^{\alpha_n}}\left(\frac{\tau_0}{\tau_n}\right)^{1/2}(2\pi)^{-n/2}

where for data mean :math:`\bar{x}` we have defined,

.. math:: \tau_n = \tau_0 + n, \quad \alpha_n = \alpha_0 + n/2, \quad b_n = b_0 +  \frac{1}{2}\sum_{i=1}^n(x_i - \bar{x})^2 + \frac{\tau_0 n(\bar{x}-\mu_0)^2}{2(\tau_0 + n)}

This therefore allows one to easily compare evidence estimates to the true evidence -- as is done here. A DAG for this problem is presented below.

.. image:: /assets/dags/hbm_normal_gamma.svg
  :width: 43 %
  :align: center

The log-likelihood function is given by

.. code-block:: python

   def ln_likelihood(x_mean, x_std, x_n, mu, tau):

    	return -0.5 * x_n * tau * (x_std**2 + (x_mean-mu)**2) - 0.5 * x_n * np.log(2 * np.pi) + 0.5 * x_n * np.log(tau)

The log-prior is given by

.. code-block:: python

   def ln_prior(mu, tau, prior_params):

   		if tau < 0:
        		return -np.inf

   		mu_0, tau_0, alpha_0, beta_0 = prior_params
   		ln_pr = alpha_0 * np.log(beta_0) + 0.5 * np.log(tau_0)
   		ln_pr += - sp.gammaln(alpha_0) - 0.5 * np.log(2*np.pi)
   		ln_pr += (alpha_0 - 0.5) * np.log(tau)
   		ln_pr += -beta_0 * tau
   		ln_pr += -0.5 * tau_0 * tau * (mu - mu_0)**2

   		return ln_pr

where the term *prior_params* is a tuple which stores the parameters :math:`\alpha_0, \beta_0, \tau_0` and :math:`\mu_0`.

We may then combine the log-likelihood and log-prior functions to define the log-posterior function simply by

.. code-block:: python
	
   def ln_posterior(theta, x_mean, x_std, x_n, prior_params):

   		mu, tau = theta
   		ln_pr = ln_prior(mu, tau, prior_params)

   		if not np.isfinite(ln_pr):
       			return -np.inf

   		ln_L = ln_likelihood(x_mean, x_std, x_n, mu, tau)

   		return  ln_L + ln_pr

Further as discussed we can explicitly calculate the analytic evidence by defining a function such as 

.. code-block:: python

   def ln_analytic_evidence(x_mean, x_std, x_n, prior_params):

   		mu_0, tau_0, alpha_0, beta_0 = prior_params
   		tau_n  = tau_0  + x_n
   		alpha_n = alpha_0 + x_n/2
   		beta_n  = beta_0 + 0.5 * x_n * x_std**2 + tau_0 * x_n * (x_mean - mu_0)**2 / (2 * (tau_0 + x_n))
   		ln_z  = sp.gammaln(alpha_n) - sp.gammaln(alpha_0)
   		ln_z += alpha_0 * np.log(beta_0) - alpha_n * np.log(beta_n)
   		ln_z += 0.5 * np.log(tau_0) - 0.5 * np.log(tau_n)
   		ln_z -= 0.5 * x_n * np.log(2*np.pi)
      
   		return ln_z
	
The first step of our evidence computation requires recovering a relatively small number of samples from the given posterior. This can be done in whatever way the user wishes, the only requirement being that a set of chains each with associated samples is provided for subsequent steps.
In our examples we choose to use the excellent `emcee  <http://dfm.io/emcee/current/>`_ python package. Utilizing emcee this example recovers samples via 

.. code-block:: python

   pos = [np.array([x_mean, 1.0/x_std**2]) + x_std * np.random.randn(ndim) /np.sqrt(x_n) for i in range(nchains)]
   sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=(x_mean, x_std, x_n, prior_params))
   rstate = np.random.get_state()
   sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
   samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
   lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

where the initial positions are drawn randomly from a uniform area of size representative of the region over which the posterior has large support.

Cross-Validation 
==========================
The cross-validation step allows **Harmonic** to compute the optimal hyper-parameter configuration for a certain class of model for a given set of posterior samples.

There are two main stages to this cross-validation process. First the MCMC chains (in this case from emcee) are configured

.. code-block:: python

   chains = hm.Chains(ndim)
   chains.add_chains_3d(samples, lnprob)
   chains_train, chains_test = hm.utils.split_data(chains, training_proportion=0.25)

before being used as training data to train a network to predict the optimal model class and optimal configuration of the hyper-parameters associated with the model class. This is done by

.. code-block:: python

   #! Make predictions for MGMM model class
   #! -------------------------------------
   validation_variances_MGMM = 
                hm.utils.cross_validation(chains_train,
                    domains_MGMM, 
                    hyper_parameters_MGMM, 
                    nfold=nfold,
                    modelClass=hm.model.ModifiedGaussianMixtureModel, 
                    seed=0)
   best_hyper_param_MGMM_ind = np.argmin(validation_variances_MGMM)
   best_hyper_param_MGMM = hyper_parameters_MGMM[best_hyper_param_MGMM_ind]

   #! Make predictions for Hyper-sphere model class
   #! ---------------------------------------------
   validation_variances_sphere = 
                hm.utils.cross_validation(chains_train,
                    domains_sphere, 
                    hyper_parameters_sphere, nfold=nfold,
                    modelClass=hm.model.HyperSphere,
                    seed=0)
   best_hyper_param_sphere_ind = np.argmin(validation_variances_sphere)
   best_hyper_param_sphere = hyper_parameters_sphere[best_hyper_param_sphere_ind]

In this case we perform cross-validation for both the MGMM and hyper-sphere model classes, from which one can select the optimal model class and the optimal set of hyper-parameters associated with that class.

Finally the now sucessfully trained network is used to make a prediction (fit) the optimal (learnt) container function :math:`\psi` -- *i.e.* the optimal hyper-parameter configuration and optimal model class -- by

.. code-block:: python

   best_var_MGMM = validation_variances_MGMM[best_hyper_param_MGMM_ind]
   best_var_sphere = validation_variances_sphere[best_hyper_param_sphere_ind]

   #! Select the optimal (minimum variance) model class
   #! -------------------------------------------------
   if best_var_MGMM < best_var_sphere:
       model = hm.model.ModifiedGaussianMixtureModel(ndim, domains_MGMM, hyper_parameters=best_hyper_param_MGMM)
       model.verbose=False
   else:
       model = hm.model.HyperSphere(ndim, domains_sphere, hyper_parameters=best_hyper_param_sphere)
   fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)

This container function is then used with the harmonic mean estimator to construct a robust computation of the Bayesian evidence by

.. code-block:: python

   ev = hm.Evidence(chains_test.nchains, model)    
   ev.add_chains(chains_test)
   ln_evidence, ln_evidence_std = ev.compute_ln_evidence()