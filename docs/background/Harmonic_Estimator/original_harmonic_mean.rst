***********************
Harmonic Mean Estimator
***********************

The harmonic mean estimator was first proposed by `Newton \& Raftery (1994) <https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/j.2517-6161.1994.tb01956.x>`_ who showed that the marginal likelihood :math:`z` can be estimated from the harmonic mean of the likelihood, given posterior samples. This follows by considering the expectation of the reciprocal of the likelihood with respect to the posterior distribution:

.. math::
	
	\rho = \mathbb{E}_{\text{P}(\theta | y)} \biggl[\frac{1}{\mathcal{L}(\theta)} \biggr] = \int \,\text{d} \theta\frac{1}{\mathcal{L}(\theta)} \text{P}(\theta | y) = \int \,\text{d} \theta \frac{1}{\mathcal{L}(\theta)} \frac{\mathcal{L}(\theta) \pi(\theta)}{z} = \frac{1}{z},

where the final line follows since the prior :math:`\pi(\theta)` is a normalised probability distribution. This relationship between the marginal likelihood and the harmonic mean motivates the **orignal harmonic mean estimator**:

.. math:: 
   \hat{\rho} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\mathcal{L}(\theta_i)}, \quad \theta_i \sim \text{P}(\theta | y),


where :math:`N` specifies the number of samples :math:`\theta_i` drawn from the posterior, and from which the evidence may naively be estimated by :math:`\hat{z} = 1 / \hat{\rho}`. For now we simply consider the estimation of the reciprocal of the evidence :math:`\hat{\rho}` and discuss estimation of the marginal likelihood and Bayes factors later.

.. warning:: As immediately realised by Neal (1994), this estimator can fail catastrophically since its variance can become very large and may not be finite. Review articles that consider a number of alternative methods to estimate the marginal likelihood have also found that the harmonic mean estimator is not robust and can be highly inaccurate, see *e.g.* `Clyde (2007) <http://adsabs.harvard.edu/pdf/2007ASPC..371..224C>`_ and `Friel (2012) <https://arxiv.org/pdf/1111.1957.pdf>`_. To understand why the estimator can lead to extremely large variance we consider an importance sampling interpretation of the harmonic mean estimator.


Importance sampling interpretation:
===================================

The harmonic mean estimator can be interpreted as importance sampling.  Consider the reciprocal marginal likelihood, which may be expressed in terms of the prior and posterior by

.. math::

	\rho = \int \,\text{d} \theta \: \frac{1}{\mathcal{L}(\theta)} \: \text{P}(\theta | y) = \int \,\text{d} \theta \: \frac{1}{z} \: \frac{\pi(\theta)}{\text{P}(\theta | y)} \: \text{P}(\theta | y).

It is clear the estimator has an importance sampling interpretation where the importance sampling target distribution is the prior :math:`\pi(\theta)`, while the sampling density is the posterior :math:`\text{P}(\theta | y)`, in contrast to typical importance sampling scenarios.

For importance sampling to be effective, one requires the sampling density to have fatter tails than the target distribution, *i.e.* to have greater probability mass in the tails of the distribution. Typically the prior has fatter tails than the posterior since the posterior updates our initial understanding of the underlying parameters :math:`\theta` that are encoded in the prior, in the presence of new data :math:`y`. For the harmonic mean estimator the importance sampling density (the posterior) typically does **not** have fatter tails than the target (the prior) and so importance sampling is not effective. This explains why the original harmonic mean estimator can be problematic. A number of variants of the original harmonic mean estimator have been intorduced in an attempt to address this issue.
