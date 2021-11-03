A standard 2D Gaussian is a simple example often used for basic benchmarking and code validation. The 2D Gaussian posterior configuration here is composed of a likelihood defined as

.. math:: \mathcal{L}(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2 + y^2}{2\sigma^2}}

and a prior defined as

.. math:: \pi(x,y) = \left\{ \begin{array}{lll}     \frac{1}{\pi w^2},  &{\rm if}  \  x^2+y^2 < w^2 \\
  		   0, \quad\quad& {\rm else} \end{array}\right.

where :math:`w` is a radial weighting function defined such that :math:`w=1` inside a circle of radius :math:`R` and 0 outside. A useful property of this particular posterior is that it permits a closed for analytic expression for the evidence, given as

.. math:: 
  :nowrap:
  
    \begin{align}
    z &= \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}dxdy\mathcal{L}(x,y)\pi(x,y),\\
		  &= \int_0^{2\pi}\int_0^w rdrd\theta\frac{1}{2\pi\sigma^2} e^{-\frac{r^2}{2\sigma^2}}\frac{1}{\pi w^2},\\
		  &= \frac{1}{\pi w^2} ( 1-e^{-\frac{w^2}{2\sigma^2}} ),\\
		  &= \frac{\beta}{\pi w^2},
    \end{align}

where we define the factor :math:`\beta = 1-e^{-\frac{w^2}{2\sigma^2}}`. Further one can compute analytic expressions for the variance and variance of the variance of the harmonic mean estimator for this particular case (see paper for details). This therefore allows one to easily compare evidence and evidence variance estimates to the true evidence and evidence variance -- as is done here.

The analytic evidence is given by

.. code-block:: python

   def ln_analytic_evidence(ndim, cov):    

    ln_norm_lik = 0.5*ndim*np.log(2*np.pi) + 0.5*np.log(np.linalg.det(cov))

    return ln_norm_lik

where the covariance is initialised by the function 

.. code-block:: python

   def init_cov(ndim):

    cov = np.zeros((ndim,ndim))
    diag_cov = np.ones(ndim) + np.random.randn(ndim)*0.1
    np.fill_diagonal(cov, diag_cov)
    
    for i in range(ndim-1):
        cov[i,i+1] = (-1)**i * 0.5*np.sqrt(cov[i,i]*cov[i+1,i+1])
        cov[i+1,i] = cov[i,i+1]
    
    return cov

The log-prior is given by the most straightforward flat uniform prior with infinite extent. We may then combine the log-likelihood and log-prior functions to define the log-posterior function simply by

.. code-block:: python
	
   def ln_posterior(x, inv_cov):
   
    return -np.dot(x,np.dot(inv_cov,x))/2.0
	
The first step of our evidence computation requires recovering a relatively small number of samples from the given posterior. This can be done in whatever way the user wishes, the only requirement being that a set of chains each with associated samples is provided for subsequent steps.
In our examples we choose to use the excellent `emcee  <http://dfm.io/emcee/current/>`_ python package. Utilizing emcee this example recovers samples via 

.. code-block:: python

   pos = np.random.rand(ndim * nchains).reshape((nchains, ndim))
   sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=[inv_cov])
   rstate = np.random.get_state()
   (pos, prob, state) = sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate) 
   samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
   lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

where the initial positions are drawn randomly from a uniform area of size representative of the region over which the posterior has large support.

As this is a Gaussian posterior the Hyper-spherical model is an obvious choice. Hence, no cross-validation is necessary and the model can be trained immediately. Having now sucessfully trained the network, we can make a prediction (fit) of the optimal (learnt) container function :math:`\psi` -- *i.e.* the optimal hyper-parameter configuration and optimal model class -- by

.. code-block:: python

   model = hm.model.HyperSphere(ndim, domains)
   fit_success, objective = model.fit(chains_train.samples, chains_train.ln_posterior)

This container function is then used with the harmonic mean estimator to construct a robust computation of the Bayesian evidence by

.. code-block:: python

   ev = hm.Evidence(chains_test.nchains, model)    
   ev.add_chains(chains_test)
   ln_evidence, ln_evidence_std = ev.compute_ln_evidence()