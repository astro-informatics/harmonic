*************************
Outputs from harmonic
*************************

**harmonic** recovers estimators for :math:`\rho` and :math:`\text{Var}[\rho]`, denoted by :math:`\hat{\rho}` and :math:`\hat{\text{Var}[\rho]}` respectively. In high-dimensional settings these values often are in danger of over- (or under-) flowing, particularly at 32 bit numerical precision. To avoid this practitioners often consider instead estimators for :math:`\log \big ( \rho \big )` upon which they recover estimates of, *e.g.* :math:`\text{Var}[\log \big ( \rho \big )]` *etc.*

The harmonic mean estimator by construction provides estimators of :math:`\rho` and is thus susceptible to these over- (or under-) flowing concerns. In **harmonic** we leverage numerical tricks, such as the `LogSumExp trick <https://en.wikipedia.org/wiki/LogSumExp>`_, to mitigate this, however this returns estimators for :math:`\log \big ( \rho \big )` and :math:`\log \big ( \text{Var}[\rho] \big )` respectively, which are **not** equivalent to their log-space counterparts. This can be straightforwardly seen by considering :math:`\log \big ( \text{Var}[\rho] \big ) \not \equiv \text{Var}[\log \big (\rho \big )]`, which follows as summation and the natural logarithm do not commute.

Given estimates of :math:`\log \big ( rho \big )` and :math:`\text{Var}[\log \big ( [\rho \big )]`, and in the situation wherein computing their exponents will immediately over- (or under-) flow, we can get a notion of how this estimated variance manifests on :math:`\rho` in log-space by considering the limiting terms :math:`x_{\pm} = \log \big (\rho \pm \sigma \big )`. One way to do this is by simply computing

.. math::

   x_{\pm} = \log \big (\rho \pm \sigma \big ) 
     = \log \Big (\rho \big ( 1 \pm \frac{\sigma}{\rho} \big ) \Big ) 
     = \log \big (\rho \big ) + \log \big ( 1 \pm \frac{\sigma}{\rho} \big ),

where we now have :math:`\eta = \sigma/\rho` which may, at first, appear to be potentially problematic. However, we can compute this ratio as a distance in log-space, normalising away any intractably large numbers, which is then almost certainly well-behaved,

.. math::

  \eta = \frac{\sigma}{\rho} = \exp \bigg ( 
  \underbrace{ 
    \log \big ( \sigma \big ) - \log \big ( \rho \big )
  }_{\text{At most $\sim \mathcal{O}(8)$}} 
  \bigg ).

Hence, :math:`x_{\pm} = \log \big ( \rho \pm \sigma \big )` is straightforward to calculate, provided the log of the estimated variance isn't further than :math:`\sim 88` from the log of the estimate of the evidence (the over/under-flow limit of the exponential function at 32 bit precision).