We present an estimator of the reciprocal marginal likelihood, an estimate of the variance of this estimator, and its variance. These estimators make use of correlated samples in order to avoid the loss of efficiency that results from thinning an MCMC chain.

We propose running a number of independent MCMC chains and using all of the correlated samples within a given chain. A number of modern MCMC sampling techniques, such as affine invariance ensemble samplers naturally provide samples from multiple chains by their ensemble nature. Moreoever, excellent software implementations are readily available, such as the `emcee <https://emcee.readthedocs.io/en/stable/>`_ code, which provides an implementation of the affine invariance ensember samplers proposed by `Goodman (2010) <https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-s.pdf>`_. Alternatively, if only a single large chain is available then this can be broken into separate blocks, which are (approximately) indpendent for a suitably long block length. Subsequently, we use the terminology chains throughout to refer to both scenarios of running multiple MCMC chains or separating a single chain in blocks.

Consider :math:`C` chains of samples, indexed by :math:`j = 1, 2, \ldots, C`, with chain :math:`j` containing :math:`N_j` samples.  The :math:`i^{\text{th}}` sample of chain :math:`j` is denoted :math:`\theta_{ij}`.  Since the chain of interest is typically clear from the context, for notational brevity we drop the chain index from the samples, *i.e.* we denotate samples by :math:`\theta_i` where the chain of interest is inferred from the context.

An estimator of the reciprocal marginal likelihood can be computed from each independent chain by

.. math::

  \hat{\rho}_j = \frac{1}{N_j} \sum_{i=1}^{N_j} \frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i) \pi(\theta_i)}, \quad \theta_i \sim \text{P}(\theta | y).

A single estimator of the reciprocal marginal likelihood can then be constructed from the estimator for each chain by

.. math::

  \hat{\rho} = \frac{\sum_{j=1}^{C} w_j \hat{\rho}_j} {\sum_{j=1}^{C} w_j },

where the estimator :math:`\hat{\rho}_j` of chain :math:`j` is weighted by the number of samples in the chain, *i.e.* :math:`w_j = N_j`.  It is straightforward to see that the estimator of the reciprocal marginal likelihood is unbiased, *i.e.* :math:`\mathbb{E}(\hat{\rho})= \rho`, since :math:`\mathbb{E}(\hat{\rho}_j) = \rho`.

The variance of the estimator :math:`\hat{\rho}` is related to the population variance :math:`\sigma^2 = \mathbb{E}\bigl[ (\hat{\rho}_i - \mathbb{E}(\hat{\rho}_i))^2 \bigr]` by

.. math::

  \text{var}(\hat{\rho}) = \frac{\sigma^2}{N_\text{eff}},

where the effective sample size is given by

.. math::

  N_\text{eff} = \frac{\bigl(\sum_j^{C} w_j \bigr)^2}{\sum_j^{C} w_j^2}.

The estimator of the population variance, given by

.. math::

  \hat{s}^2 = \frac{N_\text{eff}} {N_\text{eff}-1} \frac{\sum_{j=1}^{C} w_j (\hat{\rho}_j-\hat{\rho})^2}{\sum_j^{C} w_j},

is unbiased, *i.e.* :math:`\mathbb{E}(\hat{s}^2) = \sigma^2`. A suitable estimator for :math:`\text{var}(\hat{\rho})` is thus

.. math::

  \hat{\sigma}^2 = \frac{\hat{s}^2}{N_\text{eff}} = \frac{1} {N_\text{eff}-1} \frac{\sum_{j=1}^{C} w_j (\hat{\rho}_j-\hat{\rho})^2}{\sum_j^{C} w_j},

which is unbiased, *i.e.* :math:`\mathbb{E}(\hat{\sigma}^2) = \text{var}(\hat{\rho})`, since :math:`\hat{s}^2` is unbiased. The variance of the estimator :math:`\hat{\sigma}^2` reads

.. math::

  \text{var}(\hat{\sigma}^2) = \frac{1}{N_\text{eff}{}^2} \text{var}(\hat{s}^2) = \frac{\sigma^4}{N_\text{eff}{}^3} \biggl(\kappa - 1 + \frac{2}{N_\text{eff}-1}\biggr),

where in the second equality we have used a well-known result for the variance of the sample variance of independent and identically distributed (i.i.d.) random variables.
The kurtosis :math:`\kappa` is defined by

.. math::

  \kappa = \text{kur}(\hat{\rho}_i) = \mathbb{E} \Biggl[ \biggl(\frac{\hat{\rho}_i - \rho}{\sigma}\biggr)^4 \Biggr].

A suitable estimator for :math:`\text{var}(\hat{\sigma}^2)` is thus

.. math::

  \hat{\nu}^4 = \frac{\hat{s}^4}{N_\text{eff}{}^3} \biggl(\hat{\kappa} - 1 + \frac{2}{N_\text{eff}-1}\biggr) = \frac{\hat{\sigma}^4}{N_\text{eff}{}} \biggl(\hat{\kappa} - 1 + \frac{2}{N_\text{eff}-1}\biggr),

where for the kurtosis we adopt the estimator

.. math::

  \hat{\kappa} = \frac{\sum_{j=1}^{C} w_j (\hat{\rho}_j-\hat{\rho})^4} {\hat{s}^4 \sum_{j=1}^{C} w_j} \\ = \frac{\sum_{j=1}^{C} w_j (\hat{\rho}_j-\hat{\rho})^4} {N_\text{eff}{}^2 \hat{\sigma}^4 \sum_{j=1}^{C} w_j}.

The estimators :math:`\hat{\rho}`, :math:`\hat{\sigma}^2` and :math:`\hat{\nu}^4` provide a strategy to esimate the reciprocal marginal likelihood, its variance, and the variance of the variance, respectively. The variance estimators provide valuable measures of the accuracy of the estimated reciprocal marginal likelihood and provide useful sanity checks. Additional sanity checks can also be considered.

By the central limit theorem, for a large number of samples the distribution of :math:`\hat{\rho}_j` approaches a Gaussian, with kurtosis :math:`\kappa=3`.  If the estimated kurtosis :math:`\hat{\kappa} \gg 3` it would indicate that the sampled distribution of :math:`\hat{\rho}_j` has long tails, suggesting further samples need to be drawn.

Similarly, the ratio of :math:`\hat{\nu}^2 / \hat{\sigma}^2` can be inspected to see if it is close to that expected for a Gaussian distribution with :math:`\kappa=3` of

.. math::

  \frac{\hat{\nu}^4}{\hat{\sigma}^4} = \frac{1}{N_\text{eff}{}} \biggl(2 + \frac{2}{N_\text{eff}-1}\biggr) = \frac{2}{N_\text{eff}-1}

or equivalently

.. math::

  \frac{\hat{\nu}^2}{\hat{\sigma}^2} = \sqrt{\frac{2}{N_\text{eff}-1}}.

For the common setting where the number of samples per chain is constant, *i.e.* :math:`N_j = N` for all :math:`j`,

.. math::

  N_\text{eff} = \frac{\bigl(\sum_j^{C} w_j \bigr)^2}{\sum_j^{C} w_j^2} = \frac{(N C)^2}{N^2 C} = C

and, say :math:`C=100`, we find :math:`\hat{\nu}^2 / \hat{\sigma}^2 = 0.14`. In this setting significantly larger values of this ratio would suggest that further samples need to be drawn.