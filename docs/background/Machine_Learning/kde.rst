Kernel density estimation (KDE) provides another alternative model to learn an effective target distribution. In particular, it can be used to effectively model narrow curving posterior degeneracies.

Consider the target distribution defined by the kernel density function

.. math:: \varphi(\theta) = \frac{1}{N} \sum_i \frac{1}{V_{K}} K(\theta - \theta_i),

with kernel

.. math:: K(\theta) = k\biggl(\frac{\theta^\text{T} \Sigma_K^{-1} \theta}{R^2} \biggr),

where :math:`k(\theta) = 1` if :math:`\vert \theta \vert < 1/2` and 0 otherwise.  The volume of the kernel is given by

.. math:: V_{K} = \frac{\pi^{d/2}}{\Gamma(d/2+1)} R^d \vert \Sigma_K \vert^{1/2}.

The kernel covariance :math:`\Sigma_K` can be computed directly from the training samples, for example by estimating the covariance or even simply by the separation between the lowest and highest samples in each dimension.  A diagonal representation is often, although not always, considered for computational efficiency.

The kernel radius :math:`R` can be estimating by following a similar procedure to those outlined above for the hyper-sphere and modified Gaussian mixture model to minimise the variance of the resulting estimator. Alternatively, since there is only a single parameter cross-validation is also effective.