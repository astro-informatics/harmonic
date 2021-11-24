****************************************
Learnt Container Function
****************************************
As discussed in the harmonic mean estimator section one must select a (proper but otherwise arbitrary) **target distribution** :math:`\varphi(\theta)`. This choice is pivotal to the accuracy of the harmonic mean estimator, and so must be chosen carefully. One might initially postulate that choosing :math:`\varphi(\theta) = \pi(\theta)` would be a good choice -- where :math:`\pi(\theta)` is the prior distribution. However, provided the prior is (perhaps sensibly) not too informative, the posterior has few samples in low likelihood regions which carry very large weight, resulting in the variance of the estimator being notably large. It is then sensible to suggest functions :math:`\varphi(\theta)` such that these low likelihood are sufficiently downweighted -- *i.e.* target distributions which minimize the variance of the harmonic mean estimator. This therefore motivates the development of a **learnt models of the target distribution** which we will now discuss.

Learning from posterior samples
=======================================================

Here we cover several functional forms for the learned model :math:`\varphi(\theta)` which are used throughout the code. Hyper-parameters of these models can be considered nodes of a conventional network, the values of which are learnt from a small sub-set of posterior samples.

.. tabs::

   .. tab:: Hyper-Sphere

      .. include:: hypersphere.rst

   .. tab:: Kernel Density Estimator

      .. include:: kde.rst

   .. tab:: Modified Gaussian Mixtures

      .. include:: mgmm.rst

.. note:: This list of models is by no means comprehensive, and bespoke models may be implemented which perform better (*i.e.* lower cross-validation variance) in specific use-cases.
