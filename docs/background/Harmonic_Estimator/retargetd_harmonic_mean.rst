***********************************
Re-targeted Harmonic Mean Estimator
***********************************

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

Importance sampling interpretation:
===================================

With the introduction of the distribution :math:`\varphi(\theta)`, the importance sampling interpretation of the harmonic mean estimator reads

.. math::

	\rho = \int \,\text{d} \theta \: \frac{\varphi(\theta)}{\mathcal{L}(\theta) \pi(\theta)} \: \text{P}(\theta | y) = \int \,\text{d} \theta \: \frac{1}{z} \: \frac{\varphi(\theta)}{\text{P}(\theta | y)} \: \text{P}(\theta | y).

It is clear that the distribution :math:`\varphi(\theta)` now plays the role of the importance sampling target distribution. One is free to choose :math:`\varphi(\theta)`, with the only constraint being that it is a normalised distribution.  It is therefore possible to select the target distribution :math:`\varphi(\theta)` such that it has narrower tails than the posterior, which we recall plays the role of the importance sampling density, thereby avoiding the problematic configuration of the original harmonic mean estimator.  The selection of appropriate target densities :math:`\varphi(\theta)` for general problems remains and open question that is known to be difficult, particularly difficult in high dimensions (`Chib 1995 <https://www.jstor.org/stable/pdf/2291521.pdf?casa_token=eKhRuc2AyvMAAAAA:MY2l-YhyanFIsiQLGFDQ_T2sXjaZ60y8s5n9TicXKGGx0aZ0HtMycJ9r3nGIToRn4L0YoLXpHsl0tWeeWjZmQOpWwBFS3fheO4hk8RbIAUDLq-M9exnX>`_).