
******************************
Learnt Harmonic Mean Estimator
******************************

It is well-known that the original harmonic mean estimator can fail catastrophically since the variance of the estimator may be become very large. However, this issue can be resolved by introducing an alternative (normalised) target distribution :math:`\varphi(\theta)` (`Gelfand and Dey, 1994 <https://www.jstor.org/stable/pdf/2346123.pdf?casa_token=vIU0gg6sEy4AAAAA:21VkKf6fPFzhg2KlajF0DsILPRn4_muIES1RyGW0xaUGaPijW-YPOl5gxZIFPbadT0PeYbgcLAnuqlqSJCalFgu8W-wyrglxZCMMqzptM2sGkcG0hloG>`_), yielding what we term here the *re-targeted harmonic mean estimator*. From the importance sampling interpretation of the harmonic mean estimator, the re-targeted estimator follows by replacing the importance sampling target of the prior :math:`\pi(\theta)` with the target :math:`\varphi(\theta)`, where the posterior :math:`P(\theta | y)` plays the role of the importance sampling density.

It remains to select a suitable target distribution :math:`\varphi(\theta)`. On one hand, to ensure the variance of the resulting estimator is well-behaved, the target distribution should have narrower tails that the importance sampling density, *i.e.* the target :math:`\varphi(\theta)` should have narrower tails than the posterior :math:`P(\theta | y)`. On the other hand, to ensure the resulting estimator is efficient and makes use of as many samples from the posterior as possible, the target distribution should not be too narrow.

Optimal importance sampling target
==================================

Consider the importance sampling target distribution given by the (normalised) posterior itself:

.. math::

  \varphi^\text{optimal}(\theta) = \frac{\mathcal{L}(\theta) \pi(\theta)}{z}

This estimator is optimal in the sense of having zero variance, which is clearly apparent by substituting the target density into the re-targeted harmonic mean estimator discussed previously. Each term contributing to the summation is simply :math:`1/z`, hence the estimator :math:`\hat{\rho}` is unbiased, with zero variance.

Recall that the target density must be normalised.  Hence, the optimal estimator given by the normalised posterior is not accessible in practice since it requires the marginal likelihood -- the very term we are attempting to estimate -- to be known.  While the optimal estimator therefore cannot be used in practice, it can nevertheless be used to inform the construction of other estimators based on alternative importance sampling target distributions.

Learned the optimal sampling target
===================================

We propose learning the target distribution :math:`\varphi(\theta)` from samples of the posterior. Samples from the posterior can be split into training and evaluation (*cf.* test) sets. Machine learning (ML) techniques can then be applied to learn an approximate model of the normalised posterior from the training samples, with the constraint that the tails of the learnt target are narrower than the posterior, *i.e.*

.. math::

  \varphi(\theta) \stackrel{\text{ML}}{\simeq} \varphi^\text{optimal}(\theta) = \frac{\mathcal{L}(\theta) \pi(\theta)}{z}.

We term this approach the *learnt harmonic mean estimator*.

We are interested not only in an estimator for the marginal likelihood but also in an estimate of the variance of this estimator, and its variance. Such additional estimators are useful in their own right and can also provide valuable sanity checks that the resulting marginal likelihood estimator is well-behaved. We present corresponding estimators for the cases of uncorrelated and correlated samples.
Harmonic mean estimators provide an estimation of the reciprocal of the marginal likelihood.  We therefore also consider estimation of the marginal likelihood itself and its variance from the reciprocal estimators.  Moreover, we present expressions to also estimate the Bayes factor, and its variance, to compare two models.
Finally, we present models to learn the normalised target distribution :math:`\varphi(\theta)` by approximating the posterior distribution, with the constraint that the target has narrower tails than the posterior, and discuss how to train such models. Training involves constructing objective functions that penalise models that would result in estimators with a large variance, with appropriate regularisation.