****************************************
Machine learning
****************************************
As discussed in the harmonic mean estimator section one must select a (proper 
but otherwise arbitrary) **container function** :math:`\varphi(\theta)`. This 
choice is pivotal to the accuracy of the harmonic mean estimator, and so must be
chosen carefully. 

One might initially guess that choosing :math:`\varphi(\theta) = \pi(\theta)` 
would be a good choice -- where :math:`\pi(\theta)` is the prior distribution. 
However, provided the prior is (sensibly) not too informative, the 
posterior has few samples in low likelihood regions which carry very large 
weight, resulting in the variance of the estimator being notably large.

It is then sensible to suggest functions :math:`\varphi(\theta)` such that 
these low likelihood are sufficiently downweighted -- *i.e.* container functions
which minimize the variance of the harmonic mean estimator. This therefore 
motivates the **learnt harmonic mean estimator** which we will now discuss.

Learning :math:`\varphi(\theta)` from posterior samples
=======================================================

Here we. cover several functional forms for the container function 
:math:`\varphi(\theta)` which are used throughout the code.

Hyper-spherical Model
+++++++++++++++++++++
The first and perhaps most simple functional form is that of the uniform 
hyper-sphere.

.. math::
	:nowrap:

	\begin{equation}
	\varphi(\vect{\theta}) =\left\{ \begin{array}{lll}     \frac{1}{V_R},  &{\rm if}  ~(\vect{\theta}-\vect{\mu})^\top\vect{C}^{-1}_{\rm d}(\vect{\theta}-\vect{\mu})  < R^2 \vspace*{2mm}\\
  	0, \quad\quad& {\rm else} \end{array}\right.
  	\end{equation}
where :math:`\vect{C}_{\rm d}` is the sample covariance. If we assume 
:math:`\vect{C}_{\rm d}` to be diagonal here for computational efficiency one 
finds 

.. math::

	V_R = \frac{\pi^{N_{\rm dim}/2}}{\Gamma(N_{\rm dim}/2+1)} R^{N_{\rm dim}} \vert C_{\rm d} \vert^{1/2}.

This then simply uses the sample covariance to construct a suitable 
hyper-spherical model for a given problem.

Kernel Density Estimate
+++++++++++++++++++++++

Modified Gaussian Mixtures Model (MGMM)
+++++++++++++++++++++++++++++++++++++++
