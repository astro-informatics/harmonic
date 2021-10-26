***********************
Harmonic Mean Estimator
***********************
The harmonic mean estimator was first proposed by `Newton \& Raftery (1994) <https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/j.2517-6161.1994.tb01956.x>`_ who showed that the marginal likelihood :math:`z` can be estimated from the harmonic mean of the likelihood, given posterior samples. This follows by considering the expectation of the reciprocal of the likelihood with respect to the posterior distribution:

.. math::
	
	\rho = \mathbb{E}_{\text{P}(\theta | y)} \biggl[\frac{1}{\mathcal{L}(\theta)} \biggr] = \int \,\text{d} \theta\frac{1}{\mathcal{L}(\theta)} \text{P}(\theta | y) = \int \,\text{d} \theta \frac{1}{\mathcal{L}(\theta)} \frac{\mathcal{L}(\theta) \pi(\theta)}{z} = \frac{1}{z},

where the final line follows since the prior :math:`\pi(\theta)` is a normalised probability distribution. This relationship between the marginal likelihood and the harmonic mean motivates the **orignal harmonic mean estimator**:

.. math:: 
   \hat{\rho} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\mathcal{L}(\theta_i)}, \quad \theta_i \sim \text{P}(\theta | y),


where :math:`N` specifies the number of samples :math:`\theta_i` drawn from the posterior, and from which the evidence may naively be estimated by :math:`\hat{z} = 1 / \hat{\rho}` (for now we simply consider the estimation of the reciprocal of the evidence :math:`\hat{\rho}` and discuss estimation of the marginal likelihood and Bayes factors later.

.. warning:: As immediately realised by Neal (1994), this estimator can fail catastrophically since its variance can become very large and may not be finite. Review articles that consider a number of alternative methods to estimate the marginal likelihood have also found that the harmonic mean estimator is not accurate, *e.g.* `Clyde (2007) <http://adsabs.harvard.edu/pdf/2007ASPC..371..224C>`_ and `Friel (2012) <https://arxiv.org/pdf/1111.1957.pdf>`_. To understand why the estimator can lead to extremely large variance we consider an importance sampling interpretation of the harmonic mean estimator.


**Importance sampling interpretation:**

The harmonic mean estimator can be interpreted as importance sampling.  Consider the reciprocal marginal likelihood, which may be expressed in terms of the prior and posterior by

.. math::

	\rho = \int \,\text{d} \theta \: \frac{1}{\mathcal{L}(\theta)} \: \text{P}(\theta | y) = \int \,\text{d} \theta \: \frac{1}{z} \: \frac{\pi(\theta)}{\text{P}(\theta | y)} \: \text{P}(\theta | y).

It is clear the estimator has an importance sampling interpretation where the importance sampling target distribution is the prior :math:`\pi(\theta)`, while the sampling density is the posterior :math:`\text{P}(\theta | y)`, in contrast to typical importance sampling scenarios.

For importance sampling to be effective, one requires the sampling density to have fatter tails than the target distribution, *i.e.* to have greater probability mass in the tails of the distribution. Typically the prior has fatter tails than the posterior since the posterior updates our initial understanding of the underlying parameters :math:`\theta` that are encoded in the prior, in the presence of new data :math:`y`. For the harmonic mean estimator the importance sampling density (the posterior) typically does **not** have fatter tails than the target (the prior) and so importance sampling is not effective. This explains why the original harmonic mean estimator is problematic. A number of variants of the original harmonic mean estimator have been intorduced in an attempt to address this issue.

Re-targeted harmonic mean
====================

The original harmonic mean estimator was revised by `Gelfand (1994) <https://www.jstor.org/stable/pdf/2346123.pdf?casa_token=AB4ArghUKVEAAAAA:rEgBfQoBtpwJUFYmm07FvgnQoc9V5c07jEkctApAqlzZ1z9M16GCtDlGQsQfL5AzNgaz1YMLlN6-J7VQIy1xET9BtJyaQl_L2PEOXGjOd2MYiP7127g>`_ by introducing an arbitrary density :math:`\varphi(\theta)` to relate the reciprocal of the marginal likelihood to the likelihood through the following expectation:

.. math::

	\rho = \mathbb{E}_{\text{P}(\theta | y)} \biggl[\frac{\varphi(\theta)}{\mathcal{L}(\theta) \pi(\theta)} \biggr] = \int \,\text{d} \theta\frac{\varphi(\theta)}{\mathcal{L}(\theta) \pi(\theta)} \text{P}(\theta | y) = \int \,\text{d} \theta \frac{\varphi(\theta)}{\mathcal{L}(\theta) \pi(\theta)} \frac{\mathcal{L}(\theta) \pi(\theta)}{z} = \frac{1}{z},

where the final line follows since the density :math:`\varphi(\theta)` must be normalised. The above expression motivates the estimator:

.. math::

  \label{eqn:harmonic_mean_retargeted}
  \hat{\rho} =
  \frac{1}{N} \sum_{i=1}^N
  \frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i) \pi(\theta_i)},
  \quad
  \theta_i \sim \text{P}(\theta | y).

The normalised density :math:`\varphi(\theta)` can be interpreted as an alternative importance sampling target distribution, as we will see, hence we refer to this approach as the *re-targeted harmonic mean estimator*. Note that the original harmonic mean estimator is recovered for the target distribution :math:`\varphi(\theta) = \pi(\theta)`.

**Importance sampling interpretation:**

With the introduction of the distribution :math:`\varphi(\theta)`, the importance sampling interpretation of the harmonic mean estimator reads

.. math::

	\rho = \int \,\text{d} \theta \: \frac{\varphi(\theta)}{\mathcal{L}(\theta) \pi(\theta)} \: \text{P}(\theta | y) = \int \,\text{d} \theta \: \frac{1}{z} \: \frac{\varphi(\theta)}{\text{P}(\theta | y)} \: \text{P}(\theta | y).

It is clear that the distribution :math:`\varphi(\theta)` now plays the role of the importance sampling target distribution. One is free to choose :math:`\varphi(\theta)`, with the only constraint being that it is a normalised distribution.  It is therefore possible to select the target distribution :math:`\varphi(\theta)` such that it has narrower tails than the posterior, which we recall plays the role of the importance sampling density, thereby avoiding the problematic configuration of the original harmonic mean estimator.  However, the question of how to develop an effective strategy to select :math:`\varphi(\theta)` for a given problem remains, which is particularly difficult in high-dimensional settings.

Learnt harmonic mean
====================

It is well-known that the original harmonic mean estimator can fail catastrophically since the variance of the estimator may be become very large. However, this issue can be resolved by introducing an alternative (normalised) target distribution :math:`\varphi(\theta)`, yielding what we term here the *re-targeted harmonic mean estimator*. From the importance sampling interpretation of the harmonic mean estimator, the re-targeted estimator follows by replacing the importance sampling target of the prior :math:`\pi(\theta)` with the target :math:`\varphi(\theta)`, where the posterior :math:`P(\theta | y)` plays the role of the importance sampling density.

It remains to select a suitable target distribution :math:`\varphi(\theta)`. On one hand, to ensure the variance of the resulting estimator is well-behaved, the target distribution should have narrower tails that the importance sampling density, *i.e.* the target :math:`\varphi(\theta)` should have narrower tails than the posterior :math:`P(\theta | y)`. On the other hand, to ensure the resulting estimator is efficient and makes use of as many samples from the posterior as possible, the target distribution should not be too narrow. The optimal target distribution is the normalised posterior distribution since in this case the variance of the resulting estimator is zero. However, the normalised posterior is not accessible since it requires knowledge of the marginal likelihood, which is precisely the term we are attempting to compute.

.. centered:: We propose learning the target distribution :math:`\varphi(\theta)` from samples of the posterior. Samples from the posterior can be split into training and evaluation (*cf.* test) sets. Machine learning (ML) techniques can then be applied to learn an approximate model of the normalised posterior from the training samples, with the constraint that the tails of the learnt target are narrower than the posterior, *i.e.*

.. math::

  \varphi(\theta) \stackrel{\text{ML}}{\simeq} \varphi^\text{optimal}(\theta) = \frac{\mathcal{L}(\theta) \pi(\theta)}{z}.

.. centered:: We term this approach the *learnt harmonic mean estimator*.

We are interested not only in an estimator for the marginal likelihood but also in an estimate of the variance of this estimator, and its variance. Such additional estimators are useful in their own right and can also provide valuable sanity checks that the resulting marginal likelihood estimator is well-behaved. We present corresponding estimators for the cases of uncorrelated and correlated samples.
Harmonic mean estimators provide an estimation of the reciprocal of the marginal likelihood.  We therefore also consider estimation of the marginal likelihood itself and its variance from the reciprocal estimators.  Moreover, we present expressions to also estimate the Bayes factor, and its variance, to compare two models.
Finally, we present models to learn the normalised target distribution :math:`\varphi(\theta)` by approximating the posterior distribution, with the constraint that the target has narrower tails than the posterior, and discuss how to train such models.  Training involves constructing objective functions that penalise models that would result in estimators with a large variance, with appropriate regularisation.

.. tabs:: 
	
	.. tab:: Uncorrelated Samples

		.. include:: uncorrelated.rst

	.. tab:: Correlated Samples

		.. include:: correlated.rst

