**************************
Code Analysis
**************************

Theoretical Considerations
==========================

We consider another example where the original harmonic mean estimator was shown to fail catastrophically. In particular, we consider non-nested linear regression models for the **Radiata pine** data, which is another common benchmark data-set, and show that our learnt harmonic mean estimator is highly accurate.

For :math:`n=42` trees, the Radiata pine data-set includes measurements of the maximum compression strength parallel to the grain :math:`y_i`, density :math:`x_i` and resin-adjusted density :math:`z_i`, for specimen :math:`i \in \{1, \ldots, n\}`.  The question at hand is whether density or resin-adjusted density is a better predictor of compression strength. This motivates two Gaussian linear regression models:

.. math:: 
		:nowrap:

		\begin{align}
		&M_1 : y_i = \alpha + \beta(x_i - \bar{x}) + \epsilon_i, \epsilon_i \sim \text{N}(0, \tau^{-1}), \\
		&M_2 : y_i = \gamma + \delta(z_i - \bar{z}) + \eta_i, \eta_i \sim \text{N}(0, \lambda^{-1}),
		\end{align}

where :math:`\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i`, :math:`\bar{z} = \frac{1}{n} \sum_{i=1}^n z_i`, and :math:`\tau` and :math:`\lambda` denote the precision (inverse variance) of the noise for the respective models.


For Model 1, Gaussian priors are assumed for the bias and linear terms:

.. math:: 
	 :nowrap:

	 	\begin{align}
	 	&\alpha \sim \text{N}\bigl(\mu_\alpha, (r_0 \tau)^{-1}\bigr), \\
  		&\beta  \sim \text{N}\bigl(\mu_\beta, (s_0 \tau)^{-1}\bigr),
  		\end{align}

with means :math:`\mu_\alpha = 3000` and :math:`\mu_\beta = 185`, and precision scales :math:`r_0 = 0.06` and :math:`s_0 = 6`.  A gamma prior :math:`\tau \sim \text{Ga}(a_0, b_0)` is assumed for the noise precision with shape :math:`a_0 = 3` and rate :math:`b_0 = 2 \times 300^2`. The joint prior for :math:`(\alpha, \beta, \tau)` then reads:

.. math:: 
	:nowrap:

		\begin{align}
		\pi(\alpha, \beta, \tau) &= \pi(\alpha, \beta | \tau) \pi(\tau) = \pi(\alpha | \tau) \pi(\beta | \tau) \pi(\tau) \\   
                                 &= \frac{(b_0\tau_0)^{a_0} (r_0 s_0)^{1/2} }{2 \pi \Gamma(a_0)} \exp\bigl(-b_0 \tau\bigr) \exp\biggl(-\frac{\tau}{2}\Bigl(r_0(\alpha-\mu_\alpha)^2 + s_0(\beta-\mu_\beta)^2\Bigr)\biggr).
        \end{align}

The likelihood for Model 1 is given by

.. math:: 
	:nowrap:

		\begin{align}
		\mathcal{L}({x}, {y}) &= \prod_{i=1}^n \text{P}(x_i, y_i | \alpha, \beta, \tau), \\
                              &= \prod_{i=1}^n \sqrt{\frac{\tau}{2\pi}} \exp\Bigl(- \frac{\tau}{2} \bigl(y_i - \alpha - \beta (x_i - \bar{x})\bigr)^2\Bigr), \\
                              &= \Bigl(\frac{\tau}{2\pi}\Bigr)^{n/2} \exp\biggl(- \frac{\tau}{2} \sum_{i=1}^n \bigl(y_i - \alpha - \beta (x_i - \bar{x})\bigr)^2\biggr),
        \end{align}

where :math:`x = (x_1, \dots, x_n)^\text{T}` and :math:`y = (y_1, \dots, y_n)^\text{T}`.  For Model 2, the priors adopted for :math:`(\gamma, \delta, \lambda)` are the same as those adopted for :math:`(\alpha, \beta, \tau)` of Model 1, respectively, with the same hyperparameters.  The likelihood for Model 2 again takes an identical form to Model 1, and is presented in the DAG below.

.. image:: /assets/dags/hbm_radiata_pine.pdf
	:width: 50 %
	:align: center

Log-Likelihood, log-Prior and log-Posterior
==========================

The log-likelihood function is given by

.. code-block:: python

   def ln_likelihood(y, x, n, alpha, beta, tau):
   
    ln_like = 0.5 * n * np.log(tau)
    ln_like -= 0.5 * n * np.log(2.0 * np.pi)
    s = np.sum((y - alpha - beta * x)**2)
    ln_like -= 0.5 * tau * s
    
    return ln_like

The combined log-prior is given by

.. code-block:: python

   def ln_prior(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0):
   
    if tau < 0:
        return -np.inf

    ln_pr = a_0 * np.log(b_0)    
    ln_pr += a_0 * np.log(tau)    
    ln_pr -= b_0 * tau    
    ln_pr -= np.log(2.0 * np.pi)
    ln_pr -= sp.gammaln(a_0)
    ln_pr += 0.5 * np.log(r_0)
    ln_pr += 0.5 * np.log(s_0)
    ln_pr -= 0.5 * tau * (r_0 * (alpha - mu_0[0,0])**2 + s_0 * (beta - mu_0[1,0])**2)
    
    return ln_pr

We may then combine the log-likelihood and log-prior functions to define the log-posterior function simply by

.. code-block:: python
	
   def ln_posterior(theta, y, x, n, mu_0, r_0, s_0, a_0, b_0):
    
    alpha, beta, tau = theta
    ln_pr = ln_prior(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0)
    
    if not np.isfinite(ln_pr):
        return -np.inf

    ln_L = ln_likelihood(y, x, n, alpha, beta, tau)    
    
    return  ln_L + ln_pr

Further as discussed we can explicitly calculate the analytic evidence by defining a function such as 

.. code-block:: python

   def ln_evidence_analytic(x, y, n, mu_0, r_0, s_0, a_0, b_0):

    Q_0 = np.diag([r_0, s_0])
    X = np.c_[np.ones((n, 1)), x]
    M = X.T.dot(X) + Q_0
    nu_0 = np.linalg.inv(M).dot(X.T.dot(y) + Q_0.dot(mu_0))

    quad_terms = y.T.dot(y) + mu_0.T.dot(Q_0).dot(mu_0) - nu_0.T.dot(M).dot(nu_0)

    ln_evidence = -0.5 * n * np.log(np.pi)
    ln_evidence += a_0 * np.log(2.0*b_0)
    ln_evidence += sp.gammaln(0.5*n + a_0) - sp.gammaln(a_0)
    ln_evidence += 0.5 * np.log(np.linalg.det(Q_0)) - 0.5 * np.log(np.linalg.det(M))
    ln_evidence += -(0.5 * n + a_0) * np.log(quad_terms + 2.0 * b_0)

    return ln_evidence
	

MCMC Sampling
==========================
The first step of our evidence computation requires recovering a relatively small number of samples from the given posterior. This can be done in whatever way the user wishes, the only requirement being that a set of chains each with associated samples is provided for subsequent steps.
In our examples we choose to use the excellent `emcee  <http://dfm.io/emcee/current/>`_ python package. Utilizing emcee this example recovers samples via 

.. code-block:: python
	
   pos_alpha = mu_0[0,0] + 1.0 / np.sqrt(tau_prior_mean * r_0) * np.random.randn(nchains)  
   pos_beta = mu_0[1,0] + 1.0 / np.sqrt(tau_prior_mean * s_0) * np.random.randn(nchains)              
   pos_tau = tau_prior_mean + tau_prior_std * (np.random.rand(nchains) - 0.5)
   pos = np.c_[pos_alpha, pos_beta, pos_tau]
   
   if model_1:
       args = (y, x, n, mu_0, r_0, s_0, a_0, b_0)
   else:
       args = (y, z, n, mu_0, r_0, s_0, a_0, b_0)
   
   sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=args)
   rstate = np.random.get_state()
   sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
   samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
   lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

where the initial positions are drawn randomly from the support of each covariate prior.

Evidence estimation
==========================

We adopt the hyper-spherical model, and fit the model hyper-parameters through cross-validation as in other examples. This learnt model is then used with the harmonic mean estimator to construct a robust computation of the Bayesian evidence by

.. code-block:: python

   ev = hm.Evidence(chains_test.nchains, model)    
   ev.add_chains(chains_test)
   ln_evidence, ln_evidence_std = ev.compute_ln_evidence()





