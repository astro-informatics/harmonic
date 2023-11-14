****************************************
Learning a suitable target distribution
****************************************
While we have described estimators to compute the marginal likelihood and Bayes factors based on a learnt target distribution :math:`\varphi(\theta)`, we have yet to consider the critical task of learning the target distribution.  As discussed, the ideal target distribution is the posterior itself.  However, since the target must be normalised, use of the posterior would require knowledge of the marginal likelihood -- precisely the quantity that we attempting to estimate.  Instead,
one can learn an approximation of the posterior that is normalised.  The approximation itself does not need to be highly accurate.  More critically, the learned target approximating the posterior must exhibit narrower tails than the posterior to avoid the problematic scenario of the original harmonic mean that can result in very large variance.

We present three examples of models that can be used to learn appropriate target distributions and discuss how to train them, although other models can of course be considered.  Samples of the posterior are split into training and evaluation (*cf.* test) sets.  The training set is used to learn the target distribution, after which the evaluation set, combined with the learnt target, is used to estimate the marginal likelihood.  To train the models we typically construct and solve an optimisation problem to minimise the variance of the estimator, while ensuring it is unbiased.  We typically solve the resulting optimisation problem by stochastic gradient descent.  To set hyperparameters, we advocate cross-validation.

Learning from posterior samples
=======================================================

Here we cover several functional forms for the learned flow models :math:`\varphi(\theta)` which are used throughout the code. For these models no hyper-parameter optimisation is required, the flow should do all the heavy lifting for us!

We also provide support for legacy models, which are somewhat less expressive but nonetheless useful for simple posterior distributions. Hyper-parameters of these models can be considered nodes of a conventional network, the values of which are learnt from a small sub-set of posterior samples.

.. tabs::

   .. tab:: Normalising Flow

      .. include:: flows.rst

   .. tab:: Hyper-Sphere

      .. include:: hypersphere.rst

   .. tab:: Kernel Density Estimator

      .. include:: kde.rst

   .. tab:: Modified Gaussian Mixtures

      .. include:: mgmm.rst

.. note:: This list of models is by no means comprehensive, and bespoke models may be implemented which perform better (*i.e.* lower cross-validation variance) in specific use-cases.
