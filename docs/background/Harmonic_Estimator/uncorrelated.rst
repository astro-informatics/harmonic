MCMC algorithms that are typically used to sample the posterior distribution result in correlated samples. By suitably thinning the MCMC chain (discarding all but every :math:`t^{\text{th}}` sample), however, samples that are uncorrelated can be obtained. In this discussion we present estimators for the reciprocal marginal likelihod and its variance under the assumption of uncorrelated samples from the posterior. 

Consider the harmonic moments

.. math::

  \mu_n = \mathbb{E}_{\text{P}(\theta | y)} \Biggl[\biggl(\frac{\varphi(\theta)}{\mathcal{L}(\theta) \pi(\theta)}\biggr)^n \Biggr],

and corresponding central moments

.. math:: 

    \mu_n^\prime = \mathbb{E}_{\text{P}(\theta | y)} \Biggl[\biggl(\frac{\varphi(\theta)}{\mathcal{L}(\theta) \pi(\theta)} - \mathbb{E}_{\text{P}(\theta | y)} \biggl(\frac{\varphi(\theta)}{\mathcal{L}(\theta)\pi(\theta)} \biggr)\biggr)^n\Biggr].

We make use of the following harmonic moment estimators computed from samples of the posterior:

.. math:: 
    
    \hat{\mu}_n = \frac{1}{N} \sum_{i=1}^N \biggl(\frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i) \pi(\theta_i)}\biggr)^n,\quad \theta_i \sim \text{P}(\theta | y),

which are unbiased estimators of :math:`\mu_n`, i.e. :math:`\mathbb{E}(\hat{\mu}_n) = \mu_n`.
The reciprocal marginal likelihood can then be estimated from samples of the posterior by

.. math::

    \hat{\rho} = \hat{\mu}_1 = \frac{1}{N} \sum_{i=1}^N \frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i) \pi(\theta_i)}, \quad \theta_i \sim \text{P}(\theta | y).

The mean and variance of the estimator read, respecitvely,

.. math:: 
  
    \mathbb{E}(\hat{\rho}) = \mathbb{E} \Biggl [\frac{1}{N} \sum_{i=1}^N
    \frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i) \pi(\theta_i)} \Biggr ] = \mu_1 = \rho

and

.. math::

  \text{var}(\hat{\rho}) = \text{var} \Biggl [ \frac{1}{N} \sum_{i=1}^N \frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i) \pi(\theta_i)} \Biggr ] = \frac{1}{N} (\mu_2 - \mu_1 ^2 ).

Note that the estimator is unbiased. The optimal target is given by the normalised posterior, *i.e.* :math:`\varphi^\text{optimal}(\theta) = \mathcal{L}(\theta) \pi(\theta)/z`. It is straightforward to see that in this case

.. math::

  \mu_n = \hat{\mu}_n = \frac{1}{z^n},

and thus the target distribution is optimal since

.. math::

  \text{var}(\hat{\rho}) = \frac{1}{N} (\mu_2 - \mu_1 ^2 ) = \frac{1}{N} (1/z^2 - (1/z) ^2 ) = 0.

We are interested in not only an estimate of the reciprocal marginal likelihood but also its variance :math:`\text{var}(\hat{\rho})`. It is clear that a suitable estimator of the variance is given by

.. math::

  \hat{\sigma}^2 = \frac{1}{N-1} (\hat{\mu}_2 - \hat{\mu}_1 ^2 ) = \frac{1}{N(N-1)} \sum_{i=1}^N \biggl(\frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i) \pi(\theta_i)}\biggr)^2 - \frac{\hat{\rho}^2}{N-1}.

It follows that this estimator of the variance is unbiased since

.. math::

  \mathbb{E}(\hat{\sigma}^2) = \frac{1}{N} (\mu_2 - \mu_1^2) = \text{var}(\hat{\rho}).

The variance of the estimator :math:`\hat{\sigma}^2` reads

.. math::

  \text{var}(\hat{\sigma}^2) = \frac{1}{(N-1)^2} \biggl[ \frac{(N - 1)^2}{N^3} \mu_4^\prime - \frac{(N-1)(N-3)}{N^3} \mu_2^\prime{}^2 \biggr],

where :math:`\mu_n^\prime` are central moments, which follows by a well-known result for the variance of a sample variance, see `Rose (2002) <https://link.springer.com/chapter/10.1007/978-3-642-57489-4_66>`_ page 264. An unbiased estimator of :math:`\text{var}(\hat{\sigma}^2)` can be constructioned from h-statistics, which provide unbiased estimators of central moments.

While we have presented general estimators for uncorrelated samples here, generating uncorrelated samples requires thinning the MCMC chain, which is highly inefficient. It is generally recognised that thinning should be avoided when possible since it reduces the precision with which summaries of the MCMC chain can be computed (`Link and Eaton, 2012 <https://besjournals.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.2041-210X.2011.00131.x>`_). One may also consider estimators that do not require uncorrelated samples and so can make use of considerably more MCMC samples of the posterior.